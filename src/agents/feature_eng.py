import re
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.tools.code_executor import get_shared_sandbox
from src.tools.file_utils import save_code_to_file, save_report
from src.prompts.feature_eng_prompt import FEATURE_ENG_SYSTEM_PROMPT, FEATURE_ENG_USER_PROMPT
from src.llm_helper import call_llm_with_fallback

def feature_engineer_node(state: AgentState) -> dict:
    """
    Feature Engineering Agent: Generates and executes feature engineering code.
    
    Steps:
    1. Get cleaning results and optionally critic feedback from state
    2. Call Groq LLM to generate feature engineering code
    3. Execute in E2B sandbox (with retry on failure)
    4. Return feature code and results
    """
    print("\n" + "="*60)
    print("⚙️ FEATURE ENGINEERING AGENT")
    print("="*60)

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
⚠️ CRITICAL: The Critic Agent has reviewed your previous work and found issues.
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
            critic_section += "\n\n🔧 EXACT CODE FIXES YOU MUST APPLY:\n"
            for i, fix in enumerate(code_fixes, 1):
                critic_section += f"\n--- Fix {i}: {fix['description']} ---\n"
                critic_section += f"File: {fix['file']}\n"
                critic_section += f"Problem code:\n```python\n{fix['problem_code']}\n```\n"
                critic_section += f"Replace with:\n```python\n{fix['fixed_code']}\n```\n"
                critic_section += f"Reason: {fix['reason']}\n"
            critic_section += "\nYou MUST apply ALL fixes above and verify each one in your output."
            
        critic_section += "\n\nDo NOT repeat the same mistakes. Show that you addressed each point."
        print(f"🔄 This is iteration {iteration_count} — incorporating detailed critic feedback")
    else:
        critic_section = "This is the first pass. No critic feedback yet."

    # Call Groq LLM
    print("🤖 Calling LLM to generate feature engineering code...")

    # Build critic section if this is an iteration
    # ... (existing critic section code) ...
    
    target_col = state.get("target_column", "Churn")

    messages = [
        SystemMessage(content=FEATURE_ENG_SYSTEM_PROMPT),
        HumanMessage(content=FEATURE_ENG_USER_PROMPT.format(
            user_goal=user_goal,
            shape_after=shape_after,
            columns_after=columns_after,
            numeric_features=numeric_features,
            imbalance_ratio=imbalance_ratio,
            top_correlations=top_correlations,
            skewed_columns=skewed_columns,
            critic_section=critic_section,
            target_column=target_col,
        )),
    ]

    response, model_used = call_llm_with_fallback(messages, temperature=0.2, max_tokens=4096)
    print(f"  📡 Model used: {model_used}")
    feature_code = extract_code_block(response.content)

    if not feature_code:
        print("⚠️ No code block found, using full response")
        feature_code = response.content

    save_code_to_file(feature_code, "03_feature_engineering_code.py")
    print("✅ Feature engineering code generated")

    # Execute in shared E2B sandbox with retry
    # cleaned_data.csv already exists in sandbox from cleaner agent
    print("⚡ Executing feature engineering code in shared sandbox...")
    execution_result = ""
    success = False
    max_retries = 2

    executor = get_shared_sandbox()

    for attempt in range(max_retries + 1):
        result = executor.execute_code(feature_code)

        if result["success"]:
            execution_result = result["stdout"]
            success = True
            print(f"✅ Feature engineering executed successfully (attempt {attempt + 1})")
            
            # Verify if fixes were applied
            code_fixes = state.get("code_fixes", [])
            if iteration_count > 0 and code_fixes:
                if "FIXES APPLIED" in execution_result:
                    print("✅ Feature Engineer confirmed fixes were applied")
                else:
                    print("⚠️ Feature Engineer did not confirm fix application in output")
            break
        else:
            error_msg = result["error"]
            print(f"⚠️ Attempt {attempt + 1} failed: {error_msg[:200]}")

            if attempt < max_retries:
                print(f"🔧 Asking LLM to fix the code (retry {attempt + 1})...")
                fix_messages = messages + [
                    HumanMessage(content=f"""The code failed with this error:

{error_msg}

Fix the code and return ONLY the complete corrected Python code block.
```python
# Your fixed code here
```"""),
                ]
                fix_response, _ = call_llm_with_fallback(fix_messages, temperature=0.2, max_tokens=4096)
                feature_code = extract_code_block(fix_response.content)
                if not feature_code:
                    feature_code = fix_response.content
                save_code_to_file(feature_code, "03_feature_engineering_code.py")
            else:
                execution_result = f"FAILED after {max_retries + 1} attempts. Last error: {error_msg}"

    # Save report
    report = f"# Feature Engineering Report\n\n"
    report += f"## Status: {'SUCCESS' if success else 'FAILED'}\n"
    report += f"## Iteration: {iteration_count}\n\n"
    report += f"## Generated Code\n```python\n{feature_code}\n```\n\n"
    report += f"## Execution Output\n```\n{execution_result}\n```\n"
    save_report(report, "03_feature_engineering_report.md")

    print(f"\n📋 Feature Engineering Result Preview:")
    print(execution_result[:500] if execution_result else "No output")

    return {
        "current_agent": "feature_engineer",
        "feature_code": feature_code,
        "feature_approved": True,
        "feature_result": execution_result,
    }


def extract_code_block(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""
