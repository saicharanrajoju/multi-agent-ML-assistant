import re
import json
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.tools.code_executor import get_shared_sandbox
from src.tools.file_utils import save_code_to_file, save_report
from src.prompts.cleaner_prompt import CLEANER_SYSTEM_PROMPT, CLEANER_USER_PROMPT
from src.llm_helper import call_llm_with_fallback

def extract_code_block(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    # Fallback if no specific python block, try generic code block
    if not match:
        pattern = r"```\s*\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def cleaner_node(state: AgentState) -> dict:
    """
    Data Cleaner Agent: Generates and executes data cleaning code.
    
    Steps:
    1. Take profile report, issues, and column info from state
    2. Call Groq LLM to generate cleaning code
    3. Execute the cleaning code in E2B sandbox
    4. If execution fails, send error back to LLM to fix (max 2 retries)
    5. Return the cleaning code and execution result
    """
    print("\n" + "="*60)
    print("🧹 DATA CLEANER AGENT")
    print("="*60)

    user_goal = state["user_goal"]
    dataset_path = state["dataset_path"]
    profile_report = state.get("profile_report", "No profile available")
    data_issues = state.get("data_issues", [])
    column_info = state.get("column_info", {})

    dataset_summary = state.get("dataset_summary", {})
    target_col = state.get("target_column", "Churn")
    
    # Format dataset summary for prompt
    shape = dataset_summary.get("shape", "Unknown")
    numeric_columns = dataset_summary.get("numeric_columns", [])
    categorical_columns = dataset_summary.get("categorical_columns", [])
    binary_columns = dataset_summary.get("binary_columns", [])
    high_cardinality = dataset_summary.get("high_cardinality", [])
    missing_values = dataset_summary.get("missing_values", {})
    skewed_columns = dataset_summary.get("skewed_columns", [])

    # Format issues and column info as strings for the prompt
    issues_str = "\n".join(f"- {issue}" for issue in data_issues)
    column_info_str = "\n".join(f"- {col}: {info}" for col, info in column_info.items()) if column_info else "No column info available"

    dataset_filename = dataset_path.split("/")[-1]
    sandbox_path = f"/home/user/{dataset_filename}"

    # Call Groq LLM to generate cleaning code
    print("🤖 Calling LLM to generate cleaning code...")

    messages = [
        SystemMessage(content=CLEANER_SYSTEM_PROMPT),
        HumanMessage(content=CLEANER_USER_PROMPT.format(
            user_goal=user_goal,
            sandbox_path=sandbox_path,
            profile_report=profile_report[:3000],  # Truncate to avoid token limits
            data_issues=issues_str,
            column_info=column_info_str,
            shape=shape,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            binary_columns=binary_columns,
            high_cardinality=high_cardinality,
            missing_values=missing_values,
            skewed_columns=skewed_columns,
            target_column=target_col,
        )),
    ]

    response, model_used = call_llm_with_fallback(messages, temperature=0.1, max_tokens=4096)
    print(f"  📡 Model used: {model_used}")
    cleaning_code = extract_code_block(response.content)

    if not cleaning_code:
        print("⚠️ LLM did not return a code block, using full response")
        cleaning_code = response.content

    # Save the generated code
    save_code_to_file(cleaning_code, "02_cleaning_code.py")
    print("✅ Cleaning code generated")

    # Execute in shared E2B sandbox with retry logic
    print("⚡ Executing cleaning code in shared sandbox...")
    execution_result = ""
    success = False
    max_retries = 2

    # Dataset already uploaded by profiler — use shared sandbox
    executor = get_shared_sandbox()

    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"🔄 Retry attempt {attempt}...")

        result = executor.execute_code(cleaning_code)

        if result["success"]:
            execution_result = result["stdout"]
            success = True
            print(f"✅ Cleaning code executed successfully (attempt {attempt + 1})")
            break
        else:
            error_msg = result["error"]
            print(f"⚠️ Attempt {attempt + 1} failed: {error_msg[:200]}")

            if attempt < max_retries:
                # Ask LLM to fix the code
                print(f"🔧 Asking LLM to fix the code (retry {attempt + 1})...")
                fix_messages = messages + [
                    HumanMessage(content=f"""The code you generated failed with this error:

{error_msg}

Please fix the code and return the complete corrected version. Return ONLY the Python code block, nothing else.
```python
# Your fixed code here
```"""),
                ]
                fix_response, _ = call_llm_with_fallback(fix_messages, temperature=0.1, max_tokens=4096)
                cleaning_code = extract_code_block(fix_response.content)
                if not cleaning_code:
                    cleaning_code = fix_response.content
                save_code_to_file(cleaning_code, "02_cleaning_code.py")
            else:
                execution_result = f"FAILED after {max_retries + 1} attempts. Last error: {error_msg}"

    # Generate cleaning summary if successful
    cleaning_summary = {}
    if success:
        print("📊 Generating post-cleaning summary...")
        summary_code = """
import pandas as pd
import json

df = pd.read_csv('/home/user/cleaned_data.csv')

cleaning_summary = {
    'shape_after': list(df.shape),
    'columns_after': list(df.columns),
    'dtypes_after': {col: str(dtype) for col, dtype in df.dtypes.items()},
    'target_column': '{target_col}',
    'target_type': str(df['{target_col}'].dtype) if '{target_col}' in df.columns else 'Unknown',
    'numeric_features': list(df.select_dtypes(include='number').columns.drop('{target_col}', errors='ignore')),
    'no_missing': bool(df.isnull().sum().sum() == 0),
}
print(json.dumps(cleaning_summary))
"""
        summary_code = summary_code.replace("{target_col}", target_col)
        res = executor.execute_code(summary_code)
        if res["success"]:
            try:
                cleaning_summary = json.loads(res["stdout"])
                print("✅ Cleaning summary generated")
            except Exception as e:
                print(f"⚠️ Failed to parse cleaning summary: {e}")
        else:
            print(f"⚠️ Failed to generate cleaning summary: {res['error']}")

    # Save execution report
    report = f"# Data Cleaning Report\n\n"
    report += f"## Status: {'SUCCESS' if success else 'FAILED'}\n\n"
    report += f"## Generated Code\n```python\n{cleaning_code}\n```\n\n"
    report += f"## Execution Output\n```\n{execution_result}\n```\n"
    save_report(report, "02_cleaning_report.md")

    print(f"\n📋 Cleaning Result Preview:")
    print(execution_result[:500] if execution_result else "No output")

    return {
        "current_agent": "cleaner",
        "cleaning_code": cleaning_code,
        "cleaning_approved": True,  # Auto-approve for now, human review comes in Phase 4
        "cleaning_result": execution_result,
        "cleaning_summary": cleaning_summary,
    }
