import re
import logging
import json
import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from langchain_core.runnables import RunnableConfig
from src.tools.code_executor import get_sandbox_for_run
from src.tools.file_utils import (
    get_dataset_path, load_dataset_preview, save_report,
    extract_code_block, extract_section,
)
from src.prompts.profiler_prompt import PROFILER_SYSTEM_PROMPT, PROFILER_USER_PROMPT
from src.llm_helper import call_llm_with_fallback
from src.tools.narration import generate_narration

AGENT_TAG = "[Profiler]"

logger = logging.getLogger(__name__)

TOP_CORRELATIONS_COUNT = 5


def profiler_node(state: AgentState, config: RunnableConfig = None) -> dict:
    """
    Data Profiler Agent: Analyzes the dataset and produces a comprehensive profile.
    """
    logger.info("\n" + "="*60)
    logger.info(f"🔍 {AGENT_TAG} DATA PROFILER AGENT")
    logger.info("="*60)

    dataset_path = state["dataset_path"]
    user_goal = state["user_goal"]

    # Step 1: Get dataset preview
    logger.info(f"📊 {AGENT_TAG} Loading dataset preview...")
    full_path = get_dataset_path(dataset_path.split("/")[-1])

    try:
        df = pd.read_csv(full_path)
        preview = str(df.head())
        columns = list(df.columns)
        dtypes = str(df.dtypes)
        description = str(df.describe())
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logger.warning(f"⚠️ {AGENT_TAG} Warning: Failed to load dataset with pandas: {str(e)[:200]}")
        preview = load_dataset_preview(full_path)
        columns = []
        dtypes = ""
        description = ""

    # Step 2: Call Groq LLM
    logger.info(f"🤖 {AGENT_TAG} Calling LLM for data analysis...")

    messages = [
        SystemMessage(content=PROFILER_SYSTEM_PROMPT),
        HumanMessage(content=PROFILER_USER_PROMPT.format(
            user_goal=user_goal,
            sandbox_path="/home/user/" + dataset_path.split("/")[-1],
            dataset_preview=preview,
            columns=columns,
            dtypes=dtypes,
            description=description,
        )),
    ]

    response, model_used = call_llm_with_fallback(messages, temperature=0.1)
    logger.info(f"  📡 {AGENT_TAG} Model used: {model_used}")
    llm_output = response.content
    logger.info(f"✅ {AGENT_TAG} LLM analysis complete")

    # Step 3: Parse the response
    profile_report = extract_section(llm_output, "DATA PROFILE REPORT")
    profiling_code = extract_code_block(llm_output)
    
    # Try to parse JSON metadata first
    metadata = {}
    json_match = re.search(r'```json\s*(.*?)\s*```', llm_output, re.DOTALL | re.IGNORECASE)
    if json_match:
        try:
            metadata = json.loads(json_match.group(1))
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ {AGENT_TAG} Warning: Failed to parse JSON metadata block: {e}")
            
    if "data_issues" in metadata:
        data_issues = metadata["data_issues"]
    else:
        logger.warning(f"⚠️ {AGENT_TAG} Warning: Falling back to regex for data_issues")
        data_issues = extract_issues(llm_output)
        
    if "column_info" in metadata:
        column_info = metadata["column_info"]
    else:
        logger.warning(f"⚠️ {AGENT_TAG} Warning: Falling back to regex for column_info")
        column_info = extract_column_info(llm_output)
        
    GENERIC_FALLBACKS = {"target", "label", "output", "y", "class", ""}
    json_target = metadata.get("target_column", "").strip()
    if json_target and json_target.lower() not in GENERIC_FALLBACKS:
        target_column = json_target
    else:
        if json_target:
            logger.warning(f"⚠️ {AGENT_TAG} JSON returned generic target '{json_target}', falling back to extraction")
        target_column = extract_target_column(llm_output, user_goal)
        
    logger.info(f"🎯 {AGENT_TAG} Target column identified: {target_column}")

    # Step 4: Execute profiling code in the per-run sandbox
    run_id = (config or {}).get("configurable", {}).get("thread_id", "default")
    executor = get_sandbox_for_run(run_id)
    execution_result = ""
    if profiling_code:
        logger.info(f"⚡ {AGENT_TAG} Executing profiling code in sandbox...")
        executor.upload_file(full_path)

        result = executor.execute_code(profiling_code)
        if result["success"]:
            execution_result = result["stdout"]
            logger.info(f"✅ {AGENT_TAG} Profiling code executed successfully")
        else:
            execution_result = f"Code execution failed: {result['error']}"
            logger.warning(f"⚠️ {AGENT_TAG} Warning: Profiling code failed: {result['error'][:200]}")
    else:
        # Still upload the dataset even if no profiling code
        executor.upload_file(full_path)

    # Step 5: Combine into final report
    final_report = profile_report
    if execution_result:
        final_report += "\n\n## DETAILED PROFILING OUTPUT\n\n```\n" + execution_result + "\n```"

    # Step 5b: Generate machine-readable summary
    logger.info(f"📊 {AGENT_TAG} Generating machine-readable dataset summary...")

    sandbox_path = "/home/user/" + dataset_path.split("/")[-1]
    summary_code = """
import pandas as pd
import json

df = pd.read_csv('__SANDBOX_PATH__')
target_col = '__TARGET_COL__'

summary = {
    'shape': list(df.shape),
    'columns': list(df.columns),
    'target_column': target_col,
    'missing_values': df.isnull().sum().to_dict(),
    'numeric_columns': list(df.select_dtypes(include='number').columns),
    'categorical_columns': list(df.select_dtypes(include='object').columns),
    'target_distribution': df[target_col].value_counts().to_dict() if target_col in df.columns else {},
}

# Handle both string and numeric targets for correlation
if target_col in df.columns:
    target_series = df[target_col]
    if target_series.dtype == 'object':
        unique_vals = target_series.unique()
        mapping = {val: i for i, val in enumerate(unique_vals)}
        target_numeric = target_series.map(mapping)
    else:
        target_numeric = target_series

    correlations = {}
    for col in df.select_dtypes(include='number').columns:
        if col != target_col:
            try:
                corr = df[col].corr(target_numeric)
                if pd.notna(corr):
                    correlations[col] = round(corr, 3)
            except (TypeError, ValueError):
                pass  # column not compatible with correlation, skip

    sorted_corr = dict(sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)[:5])
    summary['correlations_with_target'] = sorted_corr

    skewed = []
    for col in df.select_dtypes(include='number').columns:
        try:
            if abs(df[col].skew()) > 1:
                skewed.append(col)
        except (TypeError, ValueError):
            pass  # non-numeric or constant column, skip safely
    summary['skewed_columns'] = skewed

    if len(df[target_col].unique()) <= 10:
        counts = df[target_col].value_counts()
        ratio = counts.min() / counts.max()
        summary['class_imbalance_ratio'] = round(ratio, 2)
    else:
        summary['class_imbalance_ratio'] = "N/A (Continuous)"

print(json.dumps(summary))
"""
    summary_code = summary_code.replace("__SANDBOX_PATH__", sandbox_path)
    summary_code = summary_code.replace("__TARGET_COL__", target_column)

    dataset_summary = {}
    res = executor.execute_code(summary_code)
    if res["success"]:
        try:
            dataset_summary = json.loads(res["stdout"])
            logger.info(f"✅ {AGENT_TAG} Machine-readable summary generated")
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ {AGENT_TAG} Warning: Failed to parse summary JSON: {str(e)[:200]}")
    else:
        logger.warning(f"⚠️ {AGENT_TAG} Warning: Failed to generate summary: {res['error'][:200]}")

    # Build reasoning context for downstream agents
    try:
        if 'df' not in locals():
            df = pd.DataFrame()
            
        problem_type, recommended_metric, reasoning_context = build_reasoning_context(
            user_goal, dataset_summary, target_column, df
        )
        logger.info(f"🧠 {AGENT_TAG} Problem type detected: {problem_type}")
        logger.info(f"🎯 {AGENT_TAG} Target metric: {recommended_metric}")
        logger.info(f"🤖 {AGENT_TAG} Recommended models: {reasoning_context['recommended_models']}")
        logger.info(f"⚖️  {AGENT_TAG} Imbalance strategy: {reasoning_context['imbalance_strategy']}")
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"⚠️ {AGENT_TAG} Warning: Could not build reasoning context: {str(e)[:200]}")
        problem_type = "binary_classification"
        recommended_metric = "f1"
        reasoning_context = {}

    # Save report
    save_report(final_report, "01_data_profile.md")

    logger.info(f"\n📋 {AGENT_TAG} Profile Report Preview:")
    logger.info(final_report[:500] + "..." if len(final_report) > 500 else final_report)
    logger.info(f"\n🔎 {AGENT_TAG} Found {len(data_issues)} data issues")

    # Generate teacher narration
    logger.info(f"📚 {AGENT_TAG} Generating teacher narration...")
    missing_cols = {k: v for k, v in dataset_summary.get("missing_values", {}).items() if v > 0}
    profiler_narration = generate_narration("profiler", {
        "target_column": target_column,
        "problem_type": problem_type,
        "recommended_metric": recommended_metric,
        "dataset_shape": dataset_summary.get("shape", []),
        "imbalance_ratio": reasoning_context.get("imbalance_ratio", "N/A"),
        "imbalance_strategy": reasoning_context.get("imbalance_strategy", "none"),
        "recommended_models": reasoning_context.get("recommended_models", []),
        "top_correlations_with_target": dataset_summary.get("correlations_with_target", {}),
        "skewed_columns": dataset_summary.get("skewed_columns", []),
        "columns_with_missing_values": missing_cols,
        "data_issues_found": data_issues[:5],
    })

    return {
        "current_agent": "profiler",
        "profile_report": final_report,
        "column_info": column_info,
        "data_issues": data_issues,
        "dataset_summary": dataset_summary,
        "target_column": target_column,
        "problem_type": problem_type,
        "recommended_metric": recommended_metric,
        "reasoning_context": reasoning_context,
        "profiler_narration": profiler_narration,
    }

def build_reasoning_context(user_goal: str, dataset_summary: dict, target_column: str, df: pd.DataFrame) -> tuple[str, str, dict]:
    """
    Analyze the dataset and user goal to produce structured reasoning for all downstream agents.
    Returns (problem_type, recommended_metric, reasoning_context)
    """
    import numpy as np
    
    # --- 1. Detect problem type ---
    # Check unique count FIRST — a 0.0/1.0 float column is still binary classification.
    problem_type = "binary_classification"  # default
    if target_column in df.columns:
        n_unique = df[target_column].nunique()
        target_dtype_str = str(df[target_column].dtype)
        if n_unique == 2:
            problem_type = "binary_classification"
        elif 2 < n_unique <= 20:
            problem_type = "multiclass_classification"
        elif target_dtype_str in ['float64', 'float32'] or n_unique > 20:
            problem_type = "regression"

    # --- 2. Parse target metric from user goal ---
    goal_lower = user_goal.lower()
    if "recall" in goal_lower or "sensitivity" in goal_lower or "false negative" in goal_lower:
        recommended_metric = "recall"
    elif "precision" in goal_lower or "false positive" in goal_lower:
        recommended_metric = "precision"
    elif "f1" in goal_lower or "f-1" in goal_lower:
        recommended_metric = "f1"
    elif "auc" in goal_lower or "roc" in goal_lower:
        recommended_metric = "roc_auc"
    elif "rmse" in goal_lower or "root mean" in goal_lower:
        recommended_metric = "rmse"
    elif "mae" in goal_lower or "mean absolute" in goal_lower:
        recommended_metric = "mae"
    elif "accuracy" in goal_lower:
        recommended_metric = "accuracy"
    else:
        # Smart default: use F1 for imbalanced classification, accuracy for balanced, RMSE for regression
        recommended_metric = "rmse" if problem_type == "regression" else "f1"

    # --- 3. Determine imbalance strategy ---
    imbalance_ratio = dataset_summary.get("class_imbalance_ratio", 1.0)
    if isinstance(imbalance_ratio, str):
        imbalance_strategy = "none"  # regression or continuous target
    elif imbalance_ratio >= 0.4:
        imbalance_strategy = "none"  # balanced enough
    elif 0.15 <= imbalance_ratio < 0.4:
        imbalance_strategy = "class_weight_balanced"
    elif 0.05 <= imbalance_ratio < 0.15:
        imbalance_strategy = "smote_plus_class_weight"
    else:
        imbalance_strategy = "threshold_tuning_plus_pr_auc"  # extreme imbalance

    # --- 4. Recommend model family based on dataset size and problem type ---
    n_rows = dataset_summary.get("shape", [0])[0] if dataset_summary.get("shape") else 0
    n_cols = dataset_summary.get("shape", [0, 0])[1] if dataset_summary.get("shape") else 0
    
    if problem_type == "regression":
        if n_rows < 500:
            recommended_models = ["LinearRegression", "Ridge", "Lasso"]
        elif n_rows < 50000:
            recommended_models = ["Ridge", "RandomForestRegressor", "XGBRegressor", "LGBMRegressor"]
        else:
            recommended_models = ["LGBMRegressor", "XGBRegressor", "Ridge"]
    else:  # classification
        if n_rows < 500:
            recommended_models = ["LogisticRegression", "RandomForestClassifier"]
        elif n_rows < 50000:
            recommended_models = ["LogisticRegression", "RandomForestClassifier", "XGBClassifier", "LGBMClassifier"]
        else:
            recommended_models = ["LGBMClassifier", "XGBClassifier", "LogisticRegression"]

    # --- 5. Detect null patterns per column ---
    null_patterns = {}
    if target_column in df.columns:
        for col in df.columns:
            if col == target_column:
                continue
            null_pct = df[col].isnull().mean()
            if null_pct == 0:
                null_patterns[col] = "none"
            elif null_pct > 0.5:
                null_patterns[col] = "high_missing_consider_drop"
            else:
                # Heuristic: if missing rate is not random (correlates with target), treat as MNAR
                null_patterns[col] = "mcar_impute"  # default; Cleaner will refine

    # --- 6. Detect encoding strategy per column ---
    encoding_map = {}
    ORDERED_ORDINALS = {
        "education": ["Preschool","1st-4th","5th-6th","7th-8th","9th","10th","11th","12th",
                      "HS-grad","Some-college","Assoc-voc","Assoc-acdm","Bachelors","Prof-school",
                      "Masters","Doctorate"],
    }
    categorical_cols = dataset_summary.get("categorical_columns", [])
    for col in categorical_cols:
        if col == target_column:
            continue
        if col.lower() in ORDERED_ORDINALS:
            encoding_map[col] = {"type": "ordinal", "order": ORDERED_ORDINALS[col.lower()]}
        elif col in df.columns and df[col].nunique() > 15:
            encoding_map[col] = "frequency_encode"
        else:
            encoding_map[col] = "onehot"

    # --- 7. Feature engineering strategies ---
    skewed_cols = dataset_summary.get("skewed_columns", [])
    feature_strategies = []
    if skewed_cols:
        feature_strategies.append(f"log_transform: {skewed_cols}")
    if imbalance_strategy in ["smote_plus_class_weight", "threshold_tuning_plus_pr_auc"]:
        feature_strategies.append("create_missing_indicator_flags_for_high_null_columns")
    if n_cols > 50:
        feature_strategies.append("remove_high_correlation_features_above_0.95")
    feature_strategies.append("create_interaction_terms_for_top_correlated_features")
    feature_strategies.append("bin_skewed_continuous_features")

    # --- 8. Detect ID-like columns (beyond just 'id' in name) ---
    _ID_PATTERNS = ['id', 'uuid', 'guid', 'key', 'hash', 'idx', 'index', 'rowid', 'row_id', 'record']
    id_columns = []
    for col in df.columns:
        if col == target_column:
            continue
        n_unique = df[col].nunique()
        col_lower = col.lower()
        is_id_name = any(
            col_lower == p or col_lower.startswith(p + '_') or col_lower.endswith('_' + p)
            for p in _ID_PATTERNS
        )
        if n_unique > 0.95 * len(df) and (is_id_name or df[col].dtype == object):
            id_columns.append(col)
        elif is_id_name and n_unique > 0.80 * len(df):
            id_columns.append(col)

    # --- 9. Detect constant / near-constant columns ---
    constant_columns = [
        col for col in df.columns
        if col != target_column and df[col].nunique() <= 1
    ]

    # --- 10. Detect datetime columns ---
    datetime_columns = []
    for col in df.columns:
        if col == target_column or col in id_columns:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_columns.append(col)
        elif df[col].dtype == object:
            sample = df[col].dropna().head(50)
            if len(sample) >= 5:
                try:
                    parsed = pd.to_datetime(sample, infer_datetime_format=True, errors='coerce')
                    if parsed.notna().mean() > 0.8:
                        full_parsed = pd.to_datetime(df[col], errors='coerce')
                        if full_parsed.notna().mean() > 0.7:
                            datetime_columns.append(col)
                except Exception:
                    pass

    # --- 11. Detect text columns (free-form, not categorical) ---
    text_columns = []
    for col in df.columns:
        if col == target_column or col in datetime_columns or col in id_columns:
            continue
        if df[col].dtype == object:
            n_unique = df[col].nunique()
            avg_len = df[col].dropna().astype(str).str.len().mean() if len(df[col].dropna()) > 0 else 0
            if n_unique > max(50, 0.3 * len(df)) and avg_len > 25:
                text_columns.append(col)

    # --- 12. Improve target ambiguity detection ---
    # Year-like integers should be regression, not multiclass
    if problem_type == "multiclass_classification" and target_column in df.columns:
        tvals = df[target_column].dropna()
        if str(tvals.dtype) in ['int64', 'int32']:
            if tvals.min() >= 1900 and tvals.max() <= 2100:
                problem_type = "regression"
            elif (tvals.max() - tvals.min()) > 50:
                problem_type = "regression"

    reasoning_context = {
        "problem_type": problem_type,
        "recommended_metric": recommended_metric,
        "imbalance_strategy": imbalance_strategy,
        "recommended_models": recommended_models,
        "null_patterns": null_patterns,
        "encoding_map": encoding_map,
        "feature_strategies": feature_strategies,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "imbalance_ratio": imbalance_ratio,
        "id_columns": id_columns,
        "constant_columns": constant_columns,
        "datetime_columns": datetime_columns,
        "text_columns": text_columns,
    }

    return problem_type, recommended_metric, reasoning_context

# --- Helper functions specific to profiler parsing ---

def extract_issues(text: str) -> list:
    """Extract data issues list from the response."""
    section = extract_section(text, "DATA ISSUES")
    issues = [line.strip().lstrip("- ").strip() for line in section.split("\n") if line.strip().startswith("-")]
    return issues if issues else ["No specific issues identified"]

def extract_column_info(text: str) -> dict:
    """Extract column information into a dict."""
    section = extract_section(text, "COLUMN INFO")

    column_info = {}
    for line in section.split("\n"):
        line = line.strip()
        if "|" in line and not line.startswith("#"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                col_name = parts[0].lstrip("- ").strip()
                if col_name:
                    column_info[col_name] = {
                        "dtype": parts[1] if len(parts) > 1 else "unknown",
                        "n_unique": parts[2] if len(parts) > 2 else "unknown",
                        "n_missing": parts[3] if len(parts) > 3 else "0",
                        "notes": parts[4] if len(parts) > 4 else "",
                    }
    return column_info


def extract_target_column(text: str, user_goal: str) -> str:
    """Extract the identified target column from profiler output."""
    # 1. Try the old markdown section format
    pattern = r"## TARGET COLUMN\s*\n\s*(\w+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()

    # 2. Look for a quoted word in the user goal — most reliable user signal
    #    e.g. predict the "diagnosis" column  →  diagnosis
    quoted = re.findall(r'["\'](\w+)["\']', user_goal)
    if quoted:
        return quoted[0]

    # 3. Look for "predict X" / "predicting X" pattern in the goal
    predict_match = re.search(r'predict(?:ing)?\s+(?:the\s+)?["\']?(\w+)["\']?', user_goal, re.IGNORECASE)
    if predict_match:
        candidate = predict_match.group(1).lower()
        # Skip generic filler words
        if candidate not in {"the", "a", "an", "whether", "if", "column", "target", "label"}:
            return predict_match.group(1)

    # 4. Keyword hardcodes for common datasets
    goal_lower = user_goal.lower()
    if "churn" in goal_lower:
        return "Churn"
    elif "surviv" in goal_lower:
        return "Survived"
    elif "fraud" in goal_lower:
        return "Class"
    elif "income" in goal_lower or "salary" in goal_lower or "earn" in goal_lower:
        return "income"
    elif "diagnosis" in goal_lower:
        return "diagnosis"
    elif "price" in goal_lower or "house" in goal_lower:
        return "price"
    elif "default" in goal_lower:
        return "default"

    return "target"  # generic fallback — will trigger validation error in graph.py
