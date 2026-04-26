import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from langchain_core.runnables import RunnableConfig
from src.tools.code_executor import get_sandbox_for_run
from src.tools.file_utils import save_code_to_file, save_report, extract_code_block, build_fix_prompt
from src.tools.code_scaffold import build_cleaning_scaffold, assemble, extract_inner_block
from src.tools.code_validator import validate_columns_against_csv
from src.prompts.cleaner_prompt import CLEANER_SYSTEM_PROMPT, CLEANER_USER_PROMPT
from src.llm_helper import call_llm_with_fallback
from src.tools.narration import generate_narration
from src.tools.pre_exec_reviewer import review_inner_block

AGENT_TAG = "[Cleaner]"

logger = logging.getLogger(__name__)

def run_post_cleaning_unit_tests(executor, target_col: str) -> dict:
    """Run unit tests on cleaned_data.csv to verify logical correctness."""
    test_code = """
import pandas as pd
import json

results = {}
try:
    df = pd.read_csv('/home/user/cleaned_data.csv')
    results['file_exists'] = True
    results['no_missing_values'] = bool(df.isnull().sum().sum() == 0)
    results['no_duplicate_columns'] = bool(len(df.columns) == len(set(df.columns)))
    results['target_column_present'] = bool('__TARGET__' in df.columns)
    results['target_is_numeric'] = bool(str(df['__TARGET__'].dtype) != 'object') if '__TARGET__' in df.columns else False
    # duplicate_row_pct is informational only — not a failure condition.
    # Real datasets often have multiple individuals with identical feature values.
    dup_count = int(df.duplicated().sum())
    results['duplicate_row_pct'] = round(dup_count / len(df) * 100, 1)
    results['row_count'] = int(len(df))
    results['all_passed'] = all([
        results['no_missing_values'],
        results['no_duplicate_columns'],
        results['target_column_present'],
        results['target_is_numeric'],
    ])
except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as e:
    results = {'file_exists': False, 'all_passed': False, 'error': str(e)}

print(json.dumps(results))
"""
    test_code = test_code.replace('__TARGET__', target_col)
    res = executor.execute_code(test_code)
    if res["success"]:
        try:
            return json.loads(res["stdout"])
        except json.JSONDecodeError:
            return {"all_passed": False, "parse_error": True}
    return {"all_passed": False, "execution_failed": True}

def cleaner_node(state: AgentState, config: RunnableConfig = None) -> dict:
    """
    Data Cleaner Agent: Generates and executes data cleaning code.
    """
    logger.info("\n" + "="*60)
    logger.info(f"🧹 {AGENT_TAG} DATA CLEANER AGENT")
    logger.info("="*60)

    user_goal = state["user_goal"]
    dataset_path = state["dataset_path"]
    profile_report = state.get("profile_report", "No profile available")
    data_issues = state.get("data_issues", [])
    column_info = state.get("column_info", {})

    dataset_summary = state.get("dataset_summary", {})
    target_col = state.get("target_column", "")
    
    reasoning_context = state.get("reasoning_context", {})
    problem_type = state.get("problem_type", "binary_classification")
    encoding_map = reasoning_context.get("encoding_map", {})
    null_patterns = reasoning_context.get("null_patterns", {})
    imbalance_strategy = reasoning_context.get("imbalance_strategy", "none")
    
    human_feedback = state.get("human_feedback", "").strip()

    # Format dataset summary for prompt
    shape = dataset_summary.get("shape", "Unknown")
    numeric_columns = dataset_summary.get("numeric_columns", [])
    categorical_columns = dataset_summary.get("categorical_columns", [])
    missing_values = dataset_summary.get("missing_values", {})
    skewed_columns = dataset_summary.get("skewed_columns", [])

    # Format issues and column info as strings for the prompt
    issues_str = "\n".join(f"- {issue}" for issue in data_issues)
    column_info_str = "\n".join(f"- {col}: {info}" for col, info in column_info.items()) if column_info else "No column info available"

    dataset_filename = dataset_path.split("/")[-1]
    sandbox_path = f"/home/user/{dataset_filename}"

    # Build human feedback section (HITL: inject reviewer instructions into the prompt)
    if human_feedback:
        human_feedback_section = (
            f"HUMAN REVIEWER INSTRUCTIONS (follow these exactly — they override default behavior):\n"
            f"{human_feedback}"
        )
        logger.info(f"📝 {AGENT_TAG} Incorporating human feedback: {human_feedback[:100]}")
    else:
        human_feedback_section = ""

    # Build scaffold — LLM only writes the business logic, not the plumbing
    scaffold_pre, scaffold_post, scaffold_instr = build_cleaning_scaffold(target_col, sandbox_path)

    logger.info(f"🤖 {AGENT_TAG} Calling LLM to generate cleaning code...")

    messages = [
        SystemMessage(content=CLEANER_SYSTEM_PROMPT),
        HumanMessage(content=CLEANER_USER_PROMPT.format(
            user_goal=user_goal,
            sandbox_path=sandbox_path,
            problem_type=problem_type,
            encoding_map=encoding_map,
            null_patterns=null_patterns,
            imbalance_strategy=imbalance_strategy,
            profile_report=profile_report[:1500],
            target_column=target_col,
            shape=shape,
            missing_values=missing_values,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            skewed_columns=skewed_columns,
            datetime_columns=reasoning_context.get("datetime_columns", []),
            id_columns=reasoning_context.get("id_columns", []),
            text_columns=reasoning_context.get("text_columns", []),
            data_issues=issues_str,
            human_feedback_section=human_feedback_section,
            scaffold_preamble=scaffold_pre,
            scaffold_instruction=scaffold_instr,
        )),
    ]

    response, model_used = call_llm_with_fallback(messages, temperature=0.1)
    logger.info(f"  📡 {AGENT_TAG} Model used: {model_used}")
    raw_block = extract_code_block(response.content) or response.content
    inner_block = extract_inner_block(raw_block, "YOUR CLEANING CODE")

    # Pre-execution review — one extra LLM call to catch critical bugs before sandbox
    reviewer_context = {
        "problem_type": problem_type,
        "target_column": target_col,
        "agent_type": "cleaner",
        "required_prints": ["SPECIAL NULLS TOTAL", "IMPUTATION", "TARGET ENCODED", "CLEANING SUMMARY"],
    }
    inner_block, pre_exec_corrections = review_inner_block(AGENT_TAG, inner_block, reviewer_context)

    cleaning_code = assemble(scaffold_pre, inner_block, scaffold_post)

    save_code_to_file(cleaning_code, "02_cleaning_code.py")
    logger.info(f"✅ {AGENT_TAG} Cleaning code generated (scaffold + LLM inner block)")

    # Execute in shared E2B sandbox with retry logic
    logger.info(f"⚡ {AGENT_TAG} Executing cleaning code in shared sandbox...")
    execution_result = ""
    success = False
    max_retries = 2

    run_id = (config or {}).get("configurable", {}).get("thread_id", "default")
    executor = get_sandbox_for_run(run_id)

    unit_test_results = {}
    for attempt in range(max_retries + 1):
        if attempt > 0:
            logger.info(f"🔄 {AGENT_TAG} Retry attempt {attempt}...")

        result = executor.execute_code(cleaning_code, timeout=120)

        if not result["success"]:
            success = False
            error_msg = result["error"]
        else:
            unit_test_results = run_post_cleaning_unit_tests(executor, target_col)
            if not unit_test_results.get("all_passed", False):
                success = False
                failed_tests = {k: v for k, v in unit_test_results.items() if v is False and k != 'all_passed'}
                error_msg = f"Local unit tests failed: {failed_tests}. Fix the logical errors in your code."
            else:
                execution_result = result["stdout"]
                success = True
                logger.info(f"✅ {AGENT_TAG} Cleaning code passed all local validation tests (attempt {attempt + 1})")
                break

        logger.warning(f"⚠️ {AGENT_TAG} Warning: Attempt {attempt + 1} failed: {error_msg[:200]}")

        if attempt < max_retries:
            logger.info(f"🔧 {AGENT_TAG} Asking LLM to fix inner block (retry {attempt + 1})...")
            fix_messages = messages + [
                HumanMessage(content=build_fix_prompt(inner_block, error_msg, attempt))
            ]
            fix_response, _ = call_llm_with_fallback(fix_messages, temperature=min(0.1 + attempt * 0.15, 0.5))
            fixed_raw = extract_code_block(fix_response.content) or fix_response.content
            inner_block = extract_inner_block(fixed_raw, "YOUR CLEANING CODE")
            cleaning_code = assemble(scaffold_pre, inner_block, scaffold_post)
            save_code_to_file(cleaning_code, "02_cleaning_code.py")
        else:
            execution_result = f"FAILED after {max_retries + 1} attempts. Last error: {error_msg}"

    # Generate cleaning summary if successful
    cleaning_summary = {}
    if success:
        logger.info(f"📊 {AGENT_TAG} Generating post-cleaning summary...")
        summary_code = """
import pandas as pd
import json

df = pd.read_csv('/home/user/cleaned_data.csv')

cleaning_summary = {
    'shape_after': list(df.shape),
    'columns_after': list(df.columns),
    'dtypes_after': {col: str(dtype) for col, dtype in df.dtypes.items()},
    'target_column': '__TARGET_COL__',
    'target_type': str(df['__TARGET_COL__'].dtype) if '__TARGET_COL__' in df.columns else 'Unknown',
    'numeric_features': list(df.select_dtypes(include='number').columns.drop('__TARGET_COL__', errors='ignore')),
    'no_missing': bool(df.isnull().sum().sum() == 0),
}
print(json.dumps(cleaning_summary))
"""
        summary_code = summary_code.replace("__TARGET_COL__", target_col)
        res = executor.execute_code(summary_code)
        if res["success"]:
            try:
                cleaning_summary = json.loads(res["stdout"])
                logger.info(f"✅ {AGENT_TAG} Cleaning summary generated")
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ {AGENT_TAG} Warning: Failed to parse cleaning summary: {str(e)[:200]}")
        else:
            logger.warning(f"⚠️ {AGENT_TAG} Warning: Failed to generate cleaning summary: {res['error'][:200]}")
    if success and unit_test_results.get("all_passed"):
        logger.info(f"✅ [Cleaner] All unit tests passed")
    elif not success and unit_test_results:
        failed = {k: v for k, v in unit_test_results.items() if v is False and k != 'all_passed'}
        logger.warning(f"⚠️ [Cleaner] Unit test failures: {failed}")

    if success:
        executor.checkpoint("cleaner")

    # Generate teacher narration (only on success)
    cleaning_narration = ""
    if success:
        logger.info(f"📚 {AGENT_TAG} Generating teacher narration...")
        missing_cols = {
            k: v for k, v in dataset_summary.get("missing_values", {}).items()
            if isinstance(v, (int, float)) and v > 0
        }
        cleaning_narration = generate_narration("cleaner", {
            "target_column": target_col,
            "original_shape": dataset_summary.get("shape", []),
            "shape_after_cleaning": cleaning_summary.get("shape_after", []),
            "columns_with_missing_values_before_cleaning": missing_cols,
            "encoding_strategy_per_column": encoding_map,
            "null_handling_per_column": null_patterns,
            "imbalance_strategy": imbalance_strategy,
            "problem_type": problem_type,
            "columns_after_cleaning": cleaning_summary.get("columns_after", [])[:15],
            "skewed_columns": skewed_columns,
        })

    # Save execution report
    report = f"# Data Cleaning Report\n\n"
    report += f"## Status: {'SUCCESS' if success else 'FAILED'}\n\n"
    report += f"## Generated Code\n```python\n{cleaning_code}\n```\n\n"
    report += f"## Execution Output\n```\n{execution_result}\n```\n"
    save_report(report, "02_cleaning_report.md")

    logger.info(f"\n📋 {AGENT_TAG} Cleaning Result Preview:")
    logger.info(execution_result[:500] if execution_result else "No output")

    return {
        "current_agent": "cleaner",
        "cleaning_code": cleaning_code,
        "cleaning_approved": success,
        "cleaning_result": execution_result,
        "cleaning_summary": cleaning_summary,
        "unit_test_results": unit_test_results,
        "cleaning_narration": cleaning_narration,
        "pre_exec_corrections": pre_exec_corrections,
        "human_feedback": "",  # clear so it doesn't bleed into downstream agents
    }
