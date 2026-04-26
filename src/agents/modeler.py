import json
import logging
import os
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from langchain_core.runnables import RunnableConfig
from src.tools.code_executor import get_sandbox_for_run
from src.tools.file_utils import save_code_to_file, save_report, extract_code_block, build_fix_prompt
from src.tools.code_scaffold import build_modeler_scaffold, assemble, extract_inner_block
from src.tools.code_validator import validate_columns_against_csv
from src.prompts.modeler_prompt import MODELER_SYSTEM_PROMPT, MODELER_USER_PROMPT
from src.llm_helper import call_llm_with_fallback
from src.tools.narration import generate_narration
from src.tools.pre_exec_reviewer import review_inner_block

AGENT_TAG = "[Modeler]"

logger = logging.getLogger(__name__)

SAMPLE_FRACTION = 0.10
MIN_ROWS_FOR_SAMPLE = 200
MAX_ROWS_FOR_SAMPLE = 2000
EXECUTION_TIMEOUT = 360



_REQUIRED_INNER_PATTERNS = [
    ("preprocessor", "ColumnTransformer/Pipeline definition"),
    ("joblib.dump", "artifact saving"),
    ("visualization_data", "visualization JSON"),
    ("best_model", "best model selection"),
    ("cross_val_score", "cross-validation"),
]

def check_inner_block_completeness(inner_block: str) -> list[str]:
    """Return list of missing required components in the modeler inner block."""
    missing = []
    for pattern, label in _REQUIRED_INNER_PATTERNS:
        if pattern not in inner_block:
            missing.append(label)
    return missing


def run_post_model_unit_tests(executor, problem_type: str) -> dict:
    """Run unit tests to verify model artifacts exist and metrics are reasonable."""
    # Metric keys differ by problem type — inject the right priority list
    if "regression" in problem_type:
        metric_keys = "['test_r2', 'r2', 'test_rmse', 'rmse', 'test_mae', 'mae']"
        # R² should be positive (better than baseline) and < 1.0
        metric_check = "metric_val > -1.0 and metric_val < 1.0"
    else:
        metric_keys = "['test_f1', 'f1', 'test_accuracy', 'accuracy', 'test_roc_auc', 'roc_auc']"
        metric_check = "metric_val > 0.0 and metric_val < 0.999"

    test_code = f"""
import os
import json

results = {{}}
try:
    model_paths = [
        '/home/user/best_model.joblib', '/home/user/best_model.pkl',
        '/home/user/model.joblib', '/home/user/model.pkl',
        '/home/user/deployment/best_model.joblib',
    ]
    preprocessor_paths = [
        '/home/user/preprocessor.joblib', '/home/user/preprocessing_pipeline.pkl',
        '/home/user/preprocessor.pkl', '/home/user/pipeline.joblib',
        '/home/user/deployment/preprocessor.joblib',
    ]

    results['model_artifact_exists'] = any(os.path.exists(p) for p in model_paths)
    results['preprocessor_artifact_exists'] = any(os.path.exists(p) for p in preprocessor_paths)
    results['metrics_file_exists'] = os.path.exists('/home/user/visualization_data.json')

    if results['metrics_file_exists']:
        with open('/home/user/visualization_data.json') as f:
            viz = json.load(f)
        metric_val = None
        for key in {metric_keys}:
            if key in viz and viz[key] is not None:
                metric_val = float(viz[key])
                break
        if metric_val is not None:
            results['metric_reasonable'] = bool({metric_check})
        else:
            results['metric_reasonable'] = None

    results['all_passed'] = all([
        results['model_artifact_exists'],
        results['preprocessor_artifact_exists'],
    ])
except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
    results = {{'all_passed': False, 'error': str(e)}}

print(json.dumps(results))
"""
    res = executor.execute_code(test_code)
    if res["success"]:
        try:
            return json.loads(res["stdout"])
        except json.JSONDecodeError:
            return {"all_passed": False, "parse_error": True}
    return {"all_passed": False, "execution_failed": True}

def run_model_scout(executor, recommended_models: list, problem_type: str, 
                    recommended_metric: str, target_col: str) -> tuple[list, list]:
    """
    Trains all recommended models on a 10% stratified sample to rank them
    empirically before full training. Returns models sorted best-to-worst.
    """
    scout_code = f"""
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

df = pd.read_csv('/home/user/featured_data.csv')

# Encode any remaining categoricals for the scout
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

target = '{target_col}'
X = df.drop(columns=[target])
y = df[target]

# Stratified 10% sample — min 200 rows, max 2000 rows
sample_size = max(200, min(2000, int(len(df) * 0.10)))
if '{problem_type}' != 'regression' and y.nunique() <= 20:
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=sample_size, stratify=y, random_state=42
    )
else:
    X_sample = X.sample(n=sample_size, random_state=42)
    y_sample = y.loc[X_sample.index]

X_sample = X_sample.fillna(X_sample.median(numeric_only=True))

results = {{}}

# Define candidate models based on problem type
if '{problem_type}' == 'regression':
    candidates = {{
        'Ridge': Ridge(),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=20, max_depth=6, random_state=42),
        'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=20, max_depth=4, random_state=42),
    }}
    if HAS_XGB:
        candidates['XGBRegressor'] = XGBRegressor(n_estimators=20, max_depth=4, random_state=42, verbosity=0)
    if HAS_LGBM:
        candidates['LGBMRegressor'] = LGBMRegressor(n_estimators=20, max_depth=4, random_state=42, verbose=-1)
    scoring = 'r2'
else:
    candidates = {{
        'LogisticRegression': LogisticRegression(max_iter=200, random_state=42),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=20, max_depth=6, random_state=42),
        'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=20, max_depth=4, random_state=42),
    }}
    if HAS_XGB:
        candidates['XGBClassifier'] = XGBClassifier(n_estimators=20, max_depth=4, random_state=42, verbosity=0, eval_metric='logloss')
    if HAS_LGBM:
        candidates['LGBMClassifier'] = LGBMClassifier(n_estimators=20, max_depth=4, random_state=42, verbose=-1)
    scoring = '{recommended_metric}' if '{recommended_metric}' in ['f1', 'roc_auc', 'precision', 'recall'] else 'f1_weighted'
    if scoring == 'f1':
        scoring = 'f1_weighted'

for name, model in candidates.items():
    try:
        pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
        scores = cross_val_score(pipe, X_sample, y_sample, cv=2, scoring=scoring)
        results[name] = round(float(scores.mean()), 4)
        print(f"SCOUT: {{name}} = {{results[name]:.4f}}")
    except (ValueError, TypeError) as e:
        results[name] = -999.0
        print(f"SCOUT: {{name}} FAILED — {{str(e)[:80]}}")

# Sort best to worst
ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
top_models = [name for name, score in ranked if score > -999]

print("SCOUT_RESULTS:" + json.dumps({{"ranked": ranked, "top_2": top_models[:2]}}))
"""
    res = executor.execute_code(scout_code)
    if res["success"]:
        for line in res["stdout"].split("\\n"):
            if line.startswith("SCOUT_RESULTS:"):
                try:
                    data = json.loads(line.replace("SCOUT_RESULTS:", ""))
                    top_2 = data.get("top_2", [])
                    ranked = data.get("ranked", [])
                    if top_2:
                        return top_2, ranked
                except json.JSONDecodeError:
                    pass
    return recommended_models, []

def modeler_node(state: AgentState, config: RunnableConfig = None) -> dict:
    """
    Model Training Agent: Generates and executes model training code.
    """
    logger.info("\n" + "="*60)
    logger.info(f"🤖 {AGENT_TAG} MODEL TRAINING AGENT")
    logger.info("="*60)

    user_goal = state["user_goal"]
    feature_result = state.get("feature_result", "No feature info available")

    dataset_summary = state.get("dataset_summary", {})
    imbalance_ratio = dataset_summary.get("class_imbalance_ratio", "Unknown")
    target_col = state.get("target_column", "")
    
    reasoning_context = state.get("reasoning_context", {})
    problem_type = state.get("problem_type", "binary_classification")
    recommended_metric = state.get("recommended_metric", "f1")
    recommended_models = reasoning_context.get("recommended_models", ["LogisticRegression", "RandomForestClassifier", "XGBClassifier", "LGBMClassifier"])
    imbalance_strategy = reasoning_context.get("imbalance_strategy", "none")
    n_rows = reasoning_context.get("n_rows", 0)
    n_cols = reasoning_context.get("n_cols", 0)

    # Check for model-specific fixes from critic
    code_fixes = state.get("code_fixes", [])
    model_fixes = [fix for fix in code_fixes if "model" in fix.get("file", "").lower()]

    model_fixes_section = ""
    if model_fixes:
        model_fixes_section = "\nCRITIC FOUND ISSUES IN YOUR PREVIOUS MODEL CODE. Apply these fixes:\n"
        for i, fix in enumerate(model_fixes, 1):
            model_fixes_section += f"\nFix {i}: {fix['description']}\n"
            model_fixes_section += f"Problem:\n```python\n{fix['problem_code']}\n```\n"
            model_fixes_section += f"Fix:\n```python\n{fix['fixed_code']}\n```\n"
            model_fixes_section += f"Reason: {fix['reason']}\n"
        logger.info(f"🔧 {AGENT_TAG} Found {len(model_fixes)} model-specific fixes to apply")

    run_id = (config or {}).get("configurable", {}).get("thread_id", "default")

    # Read actual column list from checkpointed featured_data.csv — ground truth, no guessing
    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "outputs", "checkpoints", run_id,
    )
    featured_csv_path = os.path.join(checkpoint_dir, "featured_data.csv")
    actual_feature_columns: list[str] = []
    try:
        import pandas as _pd
        actual_feature_columns = _pd.read_csv(featured_csv_path, nrows=0).columns.tolist()
        logger.info(f"📋 {AGENT_TAG} Read {len(actual_feature_columns)} actual columns from featured_data.csv")
    except Exception as _e:
        logger.warning(f"⚠️ {AGENT_TAG} Could not read featured_data.csv columns: {_e}")

    logger.info(f"🔍 {AGENT_TAG} Running model scout on 10% sample...")
    top_models, scout_ranking = run_model_scout(
        executor=get_sandbox_for_run(run_id),
        recommended_models=recommended_models,
        problem_type=problem_type,
        recommended_metric=recommended_metric,
        target_col=target_col,
    )
    logger.info(f"🏆 {AGENT_TAG} Scout selected top models: {top_models}")

    # Build scaffold — pins imports, load, X/y split, column lists, train/test split
    scaffold_pre, scaffold_post, scaffold_instr = build_modeler_scaffold(
        target_col, problem_type, top_models, n_rows
    )

    tuning_note = "Skip RandomizedSearchCV (small dataset ≤2000 rows) — refit with improved defaults" if n_rows <= 2000 else "RandomizedSearchCV n_iter=10, cv=3"
    threshold_note = "Skip threshold optimization (multiclass)" if "multiclass" in problem_type else "Threshold optimize (binary only)"
    modeling_plan = f"""- Problem type: {problem_type}
Primary metric: {recommended_metric}
SCOUT RESULTS: tested all models on 10% sample — top 2 selected: {', '.join(top_models)}
ONLY train these models in full: {', '.join(top_models)} — skip all others
Tuning: {tuning_note}
{threshold_note}
Imbalance strategy: {imbalance_strategy}
Dataset size: {n_rows} rows x {n_cols} features
Steps: Preprocessing → Train top_2 → Select best → Tune → 3-fold CV → Feature importance → Save (NO SHAP)"""
    logger.info(f"📋 {AGENT_TAG} Plan:\n{modeling_plan}")

    feature_code = state.get("feature_code", "")

    logger.info(f"🤖 {AGENT_TAG} Calling LLM to generate model training code...")

    messages = [
        SystemMessage(content=MODELER_SYSTEM_PROMPT),
        HumanMessage(content=MODELER_USER_PROMPT.format(
            user_goal=user_goal,
            problem_type=problem_type,
            recommended_metric=recommended_metric,
            recommended_models=top_models,
            imbalance_strategy=imbalance_strategy,
            n_rows=n_rows,
            n_cols=n_cols,
            imbalance_ratio=imbalance_ratio,
            feature_result=feature_result[:3000],
            target_column=target_col,
            feature_code=feature_code,
            actual_feature_columns=actual_feature_columns or "Not yet available",
            model_fixes_section=model_fixes_section,
            modeling_plan=modeling_plan,
            scaffold_preamble=scaffold_pre,
            scaffold_instruction=scaffold_instr,
        )),
    ]

    response, model_used = call_llm_with_fallback(messages, temperature=0.1)
    logger.info(f"  📡 {AGENT_TAG} Model used: {model_used}")
    raw_block = extract_code_block(response.content) or response.content
    inner_block = extract_inner_block(raw_block, "YOUR MODELING CODE")

    # Completeness check — LLM may hit output-token limit and return only a partial block
    missing_sections = check_inner_block_completeness(inner_block)
    if missing_sections:
        logger.warning(f"⚠️ {AGENT_TAG} Inner block is incomplete — missing: {missing_sections}. Requesting full rewrite...")
        completeness_msgs = [
            SystemMessage(content=MODELER_SYSTEM_PROMPT),
            HumanMessage(content=(
                MODELER_USER_PROMPT.format(
                    user_goal=user_goal,
                    problem_type=problem_type,
                    recommended_metric=recommended_metric,
                    recommended_models=top_models,
                    imbalance_strategy=imbalance_strategy,
                    n_rows=n_rows,
                    n_cols=n_cols,
                    imbalance_ratio=imbalance_ratio,
                    feature_result=feature_result[:1500],
                    target_column=target_col,
                    feature_code=feature_code[:2000],
                    actual_feature_columns=actual_feature_columns or "Not yet available",
                    model_fixes_section=model_fixes_section,
                    modeling_plan=modeling_plan,
                    scaffold_preamble=scaffold_pre,
                    scaffold_instruction=scaffold_instr,
                ) +
                f"\n\nYour previous response was INCOMPLETE — it was missing: {', '.join(missing_sections)}.\n"
                "You MUST generate a COMPLETE inner block that includes ALL of: "
                "preprocessing pipeline (ColumnTransformer), model training loop, evaluation, "
                "hyperparameter tuning, cross-validation, feature importance, "
                "joblib.dump for best_model and preprocessor, and visualization_data.json saving.\n"
                "Return the complete inner block in a single ```python code block."
            )),
        ]
        retry_resp, _ = call_llm_with_fallback(completeness_msgs, temperature=0.15)
        retry_raw = extract_code_block(retry_resp.content) or retry_resp.content
        retry_inner = extract_inner_block(retry_raw, "YOUR MODELING CODE")
        still_missing = check_inner_block_completeness(retry_inner)
        if not still_missing or len(still_missing) < len(missing_sections):
            inner_block = retry_inner
            logger.info(f"✅ {AGENT_TAG} Completeness retry improved inner block (still missing: {still_missing})")
        else:
            logger.warning(f"⚠️ {AGENT_TAG} Completeness retry did not help — proceeding with original")

    # Pre-execution review — one extra LLM call to catch critical bugs before sandbox
    reviewer_context = {
        "problem_type": problem_type,
        "target_column": target_col,
        "agent_type": "modeler",
        "required_prints": ["BEST MODEL:", "CV SCORE:", "TEST METRIC:", "PIPELINE COMPLETE:"],
    }
    inner_block, pre_exec_corrections = review_inner_block(AGENT_TAG, inner_block, reviewer_context)

    model_code = assemble(scaffold_pre, inner_block, scaffold_post)

    # Pre-execution column validator — catch hallucinated column names before sandbox
    if actual_feature_columns:
        col_warnings = validate_columns_against_csv(inner_block, featured_csv_path, target_col)
        if col_warnings:
            logger.warning(f"⚠️ {AGENT_TAG} Column validation found {len(col_warnings)} issue(s) — fixing before execution...")
            for w in col_warnings:
                logger.warning(f"   {w}")
            fix_msgs = messages + [HumanMessage(content=(
                "Your inner block references column names that don't exist in featured_data.csv. "
                f"Issues found:\n" + "\n".join(f"  - {w}" for w in col_warnings) +
                f"\n\nActual columns available: {actual_feature_columns}\n\n"
                "Fix ALL column references and return the corrected inner block only."
            ))]
            fix_resp, _ = call_llm_with_fallback(fix_msgs, temperature=0.1)
            fixed_raw = extract_code_block(fix_resp.content) or fix_resp.content
            inner_block = extract_inner_block(fixed_raw, "YOUR MODELING CODE")
            model_code = assemble(scaffold_pre, inner_block, scaffold_post)
            logger.info(f"✅ {AGENT_TAG} Column validation fix applied")

    save_code_to_file(model_code, "04_model_training_code.py")
    logger.info(f"✅ {AGENT_TAG} Model training code generated (scaffold + LLM inner block)")

    # Execute in shared E2B sandbox with retry
    logger.info(f"⚡ {AGENT_TAG} Executing model training code in shared sandbox...")
    execution_result = ""
    success = False
    MAX_RETRIES = 2

    executor = get_sandbox_for_run(run_id)

    model_unit_test_results = {}
    from src.tools.leakage_detector import detect_leakage

    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            logger.info(f"🔄 {AGENT_TAG} Retry attempt {attempt}...")

        # 1. PRE-FLIGHT CHECK: Leakage Detection
        # Scan inner_block only (not full assembled code) to avoid flagging the scaffold's own
        # LabelEncoder().fit_transform() which is intentionally before train_test_split.
        # check_split=False because the scaffold preamble always provides the split — no need to verify.
        leakage_warnings = detect_leakage(inner_block, target_col, check_split=False)
        if leakage_warnings:
            success = False
            error_msg = "AST Validation Failed (Data Leakage Detected):\n" + "\n".join(leakage_warnings)
            result = {"success": False, "error": error_msg}
        else:
            # 2. RUN CODE
            result = executor.execute_code(model_code, timeout=EXECUTION_TIMEOUT)

        if not result["success"]:
            success = False
            error_msg = result["error"]
        else:
            # 3. POST-FLIGHT LOGICAL CHECK: Unit Tests
            model_unit_test_results = run_post_model_unit_tests(executor, problem_type)
            if not model_unit_test_results.get("all_passed", False):
                success = False
                failed_tests = {k: v for k, v in model_unit_test_results.items() if v is False and k != 'all_passed'}
                error_msg = f"Local unit tests failed: {failed_tests}. Ensure your metrics, confusion matrix, and best_model.joblib are generated and saved correctly."
            else:
                execution_result = result["stdout"]
                success = True
                logger.info(f"✅ {AGENT_TAG} Model training passed all local validation (attempt {attempt + 1})")
                break

        logger.warning(f"⚠️ {AGENT_TAG} Warning: Attempt {attempt + 1} failed: {error_msg[:200]}")

        if attempt < MAX_RETRIES:
            logger.info(f"🔧 {AGENT_TAG} Asking LLM to fix inner block (retry {attempt + 1})...")
            fix_messages = messages + [
                HumanMessage(content=build_fix_prompt(inner_block, error_msg, attempt))
            ]
            fix_response, _ = call_llm_with_fallback(fix_messages, temperature=min(0.1 + attempt * 0.15, 0.5))
            fixed_raw = extract_code_block(fix_response.content) or fix_response.content
            inner_block = extract_inner_block(fixed_raw, "YOUR MODELING CODE")
            model_code = assemble(scaffold_pre, inner_block, scaffold_post)
            save_code_to_file(model_code, "04_model_training_code.py")
        else:
            execution_result = f"FAILED after {MAX_RETRIES + 1} attempts. Last error: {error_msg}"

    if success and model_unit_test_results.get("all_passed"):
        logger.info(f"✅ {AGENT_TAG} Model unit tests passed — artifacts verified")
    elif not success and model_unit_test_results:
        failed = {k: v for k, v in model_unit_test_results.items() if v is False and k != "all_passed"}
        logger.warning(f"⚠️ {AGENT_TAG} Model unit test failures: {failed}")

    # Save report
    report = f"# Model Training Report\n\n"
    report += f"## Status: {'SUCCESS' if success else 'FAILED'}\n\n"
    report += f"## Generated Code\n```python\n{model_code}\n```\n\n"
    report += f"## Training Results\n```\n{execution_result}\n```\n"
    save_report(report, "04_model_training_report.md")

    logger.info(f"\n📋 {AGENT_TAG} Model Training Result Preview:")
    logger.info(execution_result[:800] if execution_result else "No output")

    # After successful execution, try to get visualization data
    viz_data = {}
    if success:
        try:
            logger.info(f"📊 {AGENT_TAG} Retrieving visualization data...")
            viz_script = """
import json
try:
    with open('/home/user/visualization_data.json', 'r') as f:
        print(json.dumps(json.load(f)))
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error reading viz data: {e}")
"""
            viz_result = executor.execute_code(viz_script)
            if viz_result["success"] and viz_result["stdout"]:
                try:
                    output = viz_result["stdout"].strip()
                    lines = output.split('\n')
                    for line in reversed(lines):
                        if line.startswith('{') and line.endswith('}'):
                            viz_data = json.loads(line)
                            logger.info(f"📊 {AGENT_TAG} Visualization data captured!")
                            break
                    if not viz_data and output.startswith('{'):
                        viz_data = json.loads(output)
                        logger.info(f"📊 {AGENT_TAG} Visualization data captured!")
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ {AGENT_TAG} Warning: Could not parse visualization JSON: {viz_result['stdout'][:100]}")
            else:
                logger.warning(f"⚠️ {AGENT_TAG} Warning: No visualization data output found")
        except (OSError, RuntimeError) as e:
            logger.warning(f"⚠️ {AGENT_TAG} Warning: Could not load visualization data: {str(e)[:200]}")

    if success:
        executor.checkpoint("modeler")

    # Generate teacher narration (only on success, after viz_data is available)
    model_narration = ""
    if success:
        logger.info(f"📚 {AGENT_TAG} Generating teacher narration...")
        bm = viz_data.get("best_model", {})
        best_model_metrics = {
            k: v for k, v in bm.items()
            if k not in ("confusion_matrix", "feature_importance") and v is not None
        }
        cv = viz_data.get("cross_validation", {})
        model_narration = generate_narration("modeler", {
            "best_model": bm.get("name", "unknown"),
            "problem_type": problem_type,
            "recommended_metric": recommended_metric,
            "imbalance_strategy": imbalance_strategy,
            "dataset_size": f"{n_rows} rows x {n_cols} features",
            "scout_ranking_top5": scout_ranking[:5],
            "best_model_metrics": best_model_metrics,
            "cross_validation_mean": cv.get("mean"),
            "cross_validation_std": cv.get("std"),
            "models_trained": top_models,
        })

    return {
        "current_agent": "modeler",
        "model_code": model_code,
        "model_approved": success,
        "model_result": execution_result,
        "visualization_data": viz_data,
        "model_unit_test_results": model_unit_test_results,
        "scout_ranking": scout_ranking,
        "model_narration": model_narration,
        "pre_exec_corrections": pre_exec_corrections,
        "human_feedback": "",  # clear so it doesn't bleed into critic
    }
