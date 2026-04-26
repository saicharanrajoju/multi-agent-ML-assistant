import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from langchain_core.runnables import RunnableConfig
from src.tools.code_executor import get_sandbox_for_run
from src.tools.file_utils import save_code_to_file, save_report, extract_code_block, build_fix_prompt
from src.tools.code_scaffold import build_feature_eng_scaffold, assemble, extract_inner_block
from src.tools.code_validator import validate_columns_against_csv
from src.prompts.feature_eng_prompt import FEATURE_ENG_SYSTEM_PROMPT, FEATURE_ENG_USER_PROMPT
from src.llm_helper import call_llm_with_fallback
from src.tools.narration import generate_narration
from src.tools.pre_exec_reviewer import review_inner_block

AGENT_TAG = "[FeatureEng]"

logger = logging.getLogger(__name__)

def run_post_feature_unit_tests(executor, target_col: str, 
                                 original_feature_count: int) -> dict:
    """Run unit tests on featured_data.csv to verify feature engineering."""
    test_code = """
import pandas as pd
import numpy as np
import json

results = {}
try:
    df = pd.read_csv('/home/user/featured_data.csv')
    results['file_exists'] = True
    results['no_missing_values'] = bool(df.isnull().sum().sum() == 0)
    results['no_infinite_values'] = bool(not np.isinf(
        df.select_dtypes(include='number').values).any())
    results['no_duplicate_columns'] = bool(len(df.columns) == len(set(df.columns)))
    results['target_column_present'] = bool('__TARGET__' in df.columns)
    results['feature_count_grew'] = bool(len(df.columns) > __ORIG_COUNT__)
    results['all_passed'] = all([
        results['no_missing_values'],
        results['no_infinite_values'],
        results['no_duplicate_columns'],
        results['target_column_present'],
    ])
except (FileNotFoundError, ValueError, KeyError) as e:
    results = {'file_exists': False, 'all_passed': False, 'error': str(e)}

print(json.dumps(results))
"""
    test_code = test_code.replace('__TARGET__', target_col)
    test_code = test_code.replace('__ORIG_COUNT__', str(original_feature_count))
    res = executor.execute_code(test_code)
    if res["success"]:
        try:
            return json.loads(res["stdout"])
        except json.JSONDecodeError:
            return {"all_passed": False}
    return {"all_passed": False}

def feature_engineer_node(state: AgentState, config: RunnableConfig = None) -> dict:
    """
    Feature Engineering Agent: Generates and executes feature engineering code.
    """
    logger.info("\n" + "="*60)
    logger.info(f"⚙️ {AGENT_TAG} FEATURE ENGINEERING AGENT")
    logger.info("="*60)

    user_goal = state["user_goal"]
    dataset_summary = state.get("dataset_summary", {})
    cleaning_summary = state.get("cleaning_summary", {})
    iteration_count = state.get("iteration_count", 0)

    # Extract cleaning context
    shape_after = cleaning_summary.get("shape_after", "Unknown")
    columns_after = cleaning_summary.get("columns_after", [])
    numeric_features = cleaning_summary.get("numeric_features", [])

    # Extract original dataset context
    imbalance_ratio = dataset_summary.get("class_imbalance_ratio", "Unknown")
    skewed_columns = dataset_summary.get("skewed_columns", [])
    top_correlations = dataset_summary.get("correlations_with_target", {})

    # Build critic section if this is an iteration
    critic_section = ""
    if iteration_count > 0:
        critique_report = state.get("critique_report", "")
        suggestions = state.get("improvement_suggestions", [])
        scorecard = state.get("scorecard", {})

        critic_section = f"""
CRITICAL: The Critic Agent has reviewed your previous work and found issues.
You MUST address every single suggestion below. This is iteration {iteration_count}.

CRITIC'S FULL REPORT:
{critique_report}

SPECIFIC CHANGES REQUIRED (address each one):
"""
        for i, suggestion in enumerate(suggestions, 1):
            critic_section += f"\n{i}. {suggestion}"

        if scorecard:
            critic_section += f"\n\nSCORECARD FROM CRITIC:"
            critic_section += f"\n  Feature Engineering Score: {scorecard.get('feature_engineering', 'N/A')}/10"
            critic_section += f"\n  Data Leakage Score: {scorecard.get('data_leakage', 'N/A')}/10"
            critic_section += f"\n  You need to improve these scores."

        code_fixes = state.get("code_fixes", [])
        if code_fixes:
            critic_section += "\n\nEXACT CODE FIXES YOU MUST APPLY:\n"
            for i, fix in enumerate(code_fixes, 1):
                critic_section += f"\n--- Fix {i}: {fix['description']} ---\n"
                critic_section += f"File: {fix['file']}\n"
                critic_section += f"Problem code:\n```python\n{fix['problem_code']}\n```\n"
                critic_section += f"Replace with:\n```python\n{fix['fixed_code']}\n```\n"
                critic_section += f"Reason: {fix['reason']}\n"
            critic_section += "\nYou MUST apply ALL fixes above and verify each one in your output."

        critic_section += "\n\nDo NOT repeat the same mistakes. Show that you addressed each point."
        logger.info(f"🔄 {AGENT_TAG} This is iteration {iteration_count} — incorporating detailed critic feedback")
    else:
        critic_section = "This is the first pass. No critic feedback yet."

    target_col = state.get("target_column", "")
    reasoning_context = state.get("reasoning_context", {})
    problem_type = state.get("problem_type", "binary_classification")
    feature_strategies = reasoning_context.get("feature_strategies", [])

    human_feedback = state.get("human_feedback", "").strip()

    # Build human feedback section (HITL: inject reviewer instructions into the prompt)
    if human_feedback:
        human_feedback_section = (
            f"HUMAN REVIEWER INSTRUCTIONS (follow these exactly — they override default behavior):\n"
            f"{human_feedback}"
        )
        logger.info(f"📝 {AGENT_TAG} Incorporating human feedback: {human_feedback[:100]}")
    else:
        human_feedback_section = ""

    # Build scaffold — scaffold pins imports, load, X/y split, column lists, save
    scaffold_pre, scaffold_post, scaffold_instr = build_feature_eng_scaffold(
        target_col, feature_strategies, columns_after or []
    )

    logger.info(f"🤖 {AGENT_TAG} Calling LLM to generate feature engineering code...")

    messages = [
        SystemMessage(content=FEATURE_ENG_SYSTEM_PROMPT),
        HumanMessage(content=FEATURE_ENG_USER_PROMPT.format(
            user_goal=user_goal,
            problem_type=problem_type,
            feature_strategies=feature_strategies,
            top_correlations=top_correlations,
            skewed_columns=skewed_columns,
            imbalance_ratio=imbalance_ratio,
            datetime_columns=reasoning_context.get("datetime_columns", []),
            text_columns=reasoning_context.get("text_columns", []),
            shape_after=shape_after,
            columns_after=columns_after,
            numeric_features=numeric_features,
            target_column=target_col,
            critic_section=critic_section,
            human_feedback_section=human_feedback_section,
            scaffold_preamble=scaffold_pre,
            scaffold_instruction=scaffold_instr,
        )),
    ]

    response, model_used = call_llm_with_fallback(messages, temperature=0.2)
    logger.info(f"  📡 {AGENT_TAG} Model used: {model_used}")
    raw_block = extract_code_block(response.content) or response.content
    inner_block = extract_inner_block(raw_block, "YOUR FEATURE ENGINEERING CODE")

    # Pre-execution review — one extra LLM call to catch critical bugs before sandbox
    reviewer_context = {
        "problem_type": problem_type,
        "target_column": target_col,
        "agent_type": "feature_engineer",
        "required_prints": ["FEATURES BEFORE", "FEATURES AFTER", "NEW FEATURES CREATED"],
    }
    inner_block, pre_exec_corrections = review_inner_block(AGENT_TAG, inner_block, reviewer_context)

    feature_code = assemble(scaffold_pre, inner_block, scaffold_post)

    save_code_to_file(feature_code, "03_feature_engineering_code.py")
    logger.info(f"✅ {AGENT_TAG} Feature engineering code generated (scaffold + LLM inner block)")

    # Execute in shared E2B sandbox with retry
    logger.info(f"⚡ {AGENT_TAG} Executing feature engineering code in shared sandbox...")
    execution_result = ""
    success = False
    max_retries = 2

    run_id = (config or {}).get("configurable", {}).get("thread_id", "default")
    executor = get_sandbox_for_run(run_id)

    unit_test_results = {}
    from src.tools.leakage_detector import detect_leakage

    for attempt in range(max_retries + 1):
        if attempt > 0:
            logger.info(f"🔄 {AGENT_TAG} Retry attempt {attempt}...")

        # 1. PRE-FLIGHT CHECK: Leakage Detection
        leakage_warnings = detect_leakage(inner_block, target_col)
        if leakage_warnings:
            success = False
            error_msg = "AST Validation Failed (Data Leakage Detected):\n" + "\n".join(leakage_warnings)
            result = {"success": False, "error": error_msg}
        else:
            # 2. RUN CODE
            result = executor.execute_code(feature_code, timeout=120)

        if not result["success"]:
            success = False
            error_msg = result["error"]
        else:
            # 3. POST-FLIGHT LOGICAL CHECK: Unit Tests
            original_count = len(columns_after) if columns_after else 0
            unit_test_results = run_post_feature_unit_tests(executor, target_col, original_count)
            if not unit_test_results.get("all_passed", False):
                success = False
                failed_tests = {k: v for k, v in unit_test_results.items() if v is False and k != 'all_passed'}
                error_msg = f"Local unit tests failed: {failed_tests}. Fix the logical errors in your code."
            else:
                execution_result = result["stdout"]
                success = True
                logger.info(f"✅ {AGENT_TAG} Feature engineering passed all local validation (attempt {attempt + 1})")
                code_fixes = state.get("code_fixes", [])
                if iteration_count > 0 and code_fixes and "FIXES APPLIED" not in execution_result:
                    logger.warning(f"⚠️ {AGENT_TAG} Warning: Did not confirm fix application in output")
                break

        logger.warning(f"⚠️ {AGENT_TAG} Warning: Attempt {attempt + 1} failed: {error_msg[:200]}")

        if attempt < max_retries:
            logger.info(f"🔧 {AGENT_TAG} Asking LLM to fix inner block (retry {attempt + 1})...")
            fix_messages = messages + [
                HumanMessage(content=build_fix_prompt(inner_block, error_msg, attempt))
            ]
            fix_response, _ = call_llm_with_fallback(fix_messages, temperature=min(0.1 + attempt * 0.15, 0.5))
            fixed_raw = extract_code_block(fix_response.content) or fix_response.content
            inner_block = extract_inner_block(fixed_raw, "YOUR FEATURE ENGINEERING CODE")
            feature_code = assemble(scaffold_pre, inner_block, scaffold_post)
            save_code_to_file(feature_code, "03_feature_engineering_code.py")
        else:
            execution_result = f"FAILED after {max_retries + 1} attempts. Last error: {error_msg}"

    if success and unit_test_results.get("all_passed"):
        logger.info(f"✅ [FeatureEng] All unit tests passed")
    elif not success and unit_test_results:
        failed = {k: v for k, v in unit_test_results.items() if v is False and k != 'all_passed'}
        logger.warning(f"⚠️ [FeatureEng] Unit test failures: {failed}")

    if success:
        executor.checkpoint("feature_engineer")

    # Generate teacher narration (only on success)
    feature_narration = ""
    if success:
        logger.info(f"📚 {AGENT_TAG} Generating teacher narration...")
        feature_narration = generate_narration("feature_eng", {
            "problem_type": problem_type,
            "target_column": target_col,
            "feature_strategies_applied": feature_strategies,
            "top_correlations_with_target": top_correlations,
            "skewed_columns_found": skewed_columns,
            "class_imbalance_ratio": imbalance_ratio,
            "columns_entering_feature_engineering": columns_after[:15] if columns_after else [],
            "dataset_shape_before": shape_after,
        })

    # Save report
    report = f"# Feature Engineering Report\n\n"
    report += f"## Status: {'SUCCESS' if success else 'FAILED'}\n"
    report += f"## Iteration: {iteration_count}\n\n"
    report += f"## Generated Code\n```python\n{feature_code}\n```\n\n"
    report += f"## Execution Output\n```\n{execution_result}\n```\n"
    save_report(report, "03_feature_engineering_report.md")

    logger.info(f"\n📋 {AGENT_TAG} Feature Engineering Result Preview:")
    logger.info(execution_result[:500] if execution_result else "No output")

    return {
        "current_agent": "feature_engineer",
        "feature_code": feature_code,
        "feature_approved": success,
        "feature_result": execution_result,
        "unit_test_results": unit_test_results,
        "feature_narration": feature_narration,
        "pre_exec_corrections": pre_exec_corrections,
        "human_feedback": "",  # clear so it doesn't bleed into downstream agents
    }
