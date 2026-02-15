import re
import json
import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.tools.code_executor import get_shared_sandbox
from src.tools.file_utils import get_dataset_path, load_dataset_preview, save_report
from src.prompts.profiler_prompt import PROFILER_SYSTEM_PROMPT, PROFILER_USER_PROMPT
from src.llm_helper import call_llm_with_fallback

def profiler_node(state: AgentState) -> dict:
    """
    Data Profiler Agent: Analyzes the dataset and produces a comprehensive profile.
    
    Steps:
    1. Load a preview of the dataset locally (for the LLM prompt)
    2. Send preview + goal to Groq LLM with profiler prompt
    3. Parse the LLM response to extract report, code, issues, and column info
    4. Execute the profiling code in E2B sandbox for detailed stats
    5. Combine LLM analysis + execution results into final profile
    6. Return state updates
    """
    print("\n" + "="*60)
    print("🔍 DATA PROFILER AGENT")
    print("="*60)

    dataset_path = state["dataset_path"]
    user_goal = state["user_goal"]

    # Step 1: Get dataset preview
    print("📊 Loading dataset preview...")
    full_path = get_dataset_path(dataset_path.split("/")[-1])
    
    # Load full dataset to get more info for prompt
    try:
        df = pd.read_csv(full_path)
        preview = str(df.head())
        columns = list(df.columns)
        dtypes = str(df.dtypes)
        description = str(df.describe())
    except Exception as e:
        print(f"⚠️ Failed to load dataset with pandas: {str(e)}")
        # Fallback to simple preview if pandas fails
        preview = load_dataset_preview(full_path)
        columns = []
        dtypes = ""
        description = ""


    # Step 2: Call Groq LLM
    print("🤖 Calling LLM for data analysis...")

    messages = [
        SystemMessage(content=PROFILER_SYSTEM_PROMPT),
        HumanMessage(content=PROFILER_USER_PROMPT.format(
            user_goal=user_goal,
            sandbox_path="/home/user/" + dataset_path.split("/")[-1],
            dataset_preview=preview,
            columns=columns, # Pass columns
            dtypes=dtypes,   # Pass dtypes
            description=description # Pass description
        )),
    ]

    response, model_used = call_llm_with_fallback(messages, temperature=0.1, max_tokens=4096)
    print(f"  📡 Model used: {model_used}")
    llm_output = response.content
    print("✅ LLM analysis complete")

    # Step 3: Parse the response
    profile_report = extract_section(llm_output, "DATA PROFILE REPORT")
    profiling_code = extract_code_block(llm_output)
    data_issues = extract_issues(llm_output)
    column_info = extract_column_info(llm_output)
    target_column = extract_target_column(llm_output, user_goal) # Extract target column
    print(f"🎯 Target column identified: {target_column}")

    # Step 4: Execute profiling code in shared E2B sandbox
    execution_result = ""
    if profiling_code:
        print("⚡ Executing profiling code in sandbox...")
        executor = get_shared_sandbox()
        # Upload dataset to shared sandbox (first agent to do so)
        executor.upload_file(full_path)

        result = executor.execute_code(profiling_code)
        if result["success"]:
            execution_result = result["stdout"]
            print("✅ Profiling code executed successfully")
        else:
            execution_result = f"Code execution failed: {result['error']}"
            print(f"⚠️ Profiling code failed: {result['error'][:200]}")

    # Step 5: Combine into final report
    final_report = profile_report
    if execution_result:
        final_report += "\n\n## DETAILED PROFILING OUTPUT\n\n```\n" + execution_result + "\n```"

    # Step 5b: Generate machine-readable summary
    print("📊 Generating machine-readable dataset summary...")
    
    # We pass the target_column python variable into the f-string
    summary_code = f"""
import pandas as pd
import json

df = pd.read_csv('{{sandbox_path}}')
target_col = '{target_column}'

summary = {{
    'shape': list(df.shape),
    'columns': list(df.columns),
    'target_column': target_col,
    'missing_values': df.isnull().sum().to_dict(),
    'numeric_columns': list(df.select_dtypes(include='number').columns),
    'categorical_columns': list(df.select_dtypes(include='object').columns),
    'target_distribution': df[target_col].value_counts().to_dict() if target_col in df.columns else {{}},
}}

# Handle both string and numeric targets for correlation
if target_col in df.columns:
    target_series = df[target_col]
    # If target is categorical/object, map it to numbers for correlation
    if target_series.dtype == 'object':
        # Create a mapping based on value counts or simple enumeration
        # For binary classification (like Yes/No), this often works well enough for rough correlation
        unique_vals = target_series.unique()
        mapping = {{val: i for i, val in enumerate(unique_vals)}}
        target_numeric = target_series.map(mapping)
    else:
        target_numeric = target_series
    
    correlations = {{}}
    for col in df.select_dtypes(include='number').columns:
        if col != target_col:
            try:
                # Basic correlation
                corr = df[col].corr(target_numeric)
                if pd.notna(corr):
                    correlations[col] = round(corr, 3)
            except:
                pass
                
    # Sort correlations by absolute value
    sorted_corr = dict(sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)[:5])
    summary['correlations_with_target'] = sorted_corr
    
    # Identify skewed columns (skew > 1)
    skewed = []
    for col in df.select_dtypes(include='number').columns:
        try:
            if abs(df[col].skew()) > 1:
                skewed.append(col)
        except:
             pass
    summary['skewed_columns'] = skewed

    # Calculate class imbalance
    if len(df[target_col].unique()) <= 10:
        counts = df[target_col].value_counts()
        ratio = counts.min() / counts.max()
        summary['class_imbalance_ratio'] = round(ratio, 2)
    else:
        summary['class_imbalance_ratio'] = "N/A (Continuous)"

print(json.dumps(summary))
"""
    sandbox_path = "/home/user/" + dataset_path.split("/")[-1]
    # Replace sandbox path placeholder
    # Note: We use .replace for sandbox_path to avoid conflicts with the inner JSON structure
    summary_code = summary_code.replace("{sandbox_path}", sandbox_path)
    
    dataset_summary = {}
    if profiling_code:  # Only if we can execute code
        # Re-use existing executor if available, otherwise get new one
        if 'executor' not in locals():
             executor = get_shared_sandbox()
             
        res = executor.execute_code(summary_code)
        if res["success"]:
            try:
                dataset_summary = json.loads(res["stdout"])
                print("✅ Machine-readable summary generated")
            except Exception as e:
                print(f"⚠️ Failed to parse summary JSON: {e}")
        else:
             print(f"⚠️ Failed to generate summary: {res['error']}")

    # Save report
    save_report(final_report, "01_data_profile.md")

    print("\n📋 Profile Report Preview:")
    print(final_report[:500] + "..." if len(final_report) > 500 else final_report)
    print(f"\n🔎 Found {len(data_issues)} data issues")

    return {
        "current_agent": "profiler",
        "profile_report": final_report,
        "column_info": column_info,
        "data_issues": data_issues,
        "dataset_summary": dataset_summary,
        "target_column": target_column,
    }


# --- Helper functions to parse LLM output ---

def extract_section(text: str, start_header: str, end_header: str = None) -> str:
    """Extract content between headers."""
    if end_header:
        pattern = rf"## {start_header}\s*\n(.*?)(?=## {end_header}|$)"
    else:
        # Default behavior: extract until next ## header or end of string
        pattern = rf"## {start_header}\s*\n(.*?)(?=## |$)"
        
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text

def extract_code_block(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

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
    # Try to find explicit target section
    pattern = r"## TARGET COLUMN\s*\n\s*(\w+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    
    # Fallback: guess from user goal
    goal_lower = user_goal.lower()
    if "churn" in goal_lower:
        return "Churn"
    elif "surviv" in goal_lower:
        return "Survived"
    elif "fraud" in goal_lower:
        return "Class"
    
    return "target"  # generic fallback
