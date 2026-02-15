import re
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.tools.code_executor import get_shared_sandbox
from src.tools.file_utils import save_code_to_file, save_report
from src.prompts.modeler_prompt import MODELER_SYSTEM_PROMPT, MODELER_USER_PROMPT
from src.llm_helper import call_llm_with_fallback

def modeler_node(state: AgentState) -> dict:
    """
    Model Training Agent: Generates and executes model training code.
    
    Steps:
    1. Get feature engineering results from state
    2. Call Groq LLM to generate model training code
    3. Re-run cleaning + feature engineering in sandbox to prepare data
    4. Execute model training code in sandbox
    5. Retry on failure (max 2 retries)
    6. Return model code and results (metrics, best model info)
    """
    print("\n" + "="*60)
    print("🤖 MODEL TRAINING AGENT")
    print("="*60)

    user_goal = state["user_goal"]
    feature_result = state.get("feature_result", "No feature info available")
    cleaning_code = state.get("cleaning_code", "")
    feature_code = state.get("feature_code", "")

    dataset_summary = state.get("dataset_summary", {})
    imbalance_ratio = dataset_summary.get("class_imbalance_ratio", "Unknown")
    target_col = state.get("target_column", "Churn")

    # Check for model-specific fixes from critic
    code_fixes = state.get("code_fixes", [])
    model_fixes = [fix for fix in code_fixes if "model" in fix.get("file", "").lower()]
    
    model_fixes_section = ""
    if model_fixes:
        model_fixes_section = "\n⚠️ CRITIC FOUND ISSUES IN YOUR PREVIOUS MODEL CODE. Apply these fixes:\n"
        for i, fix in enumerate(model_fixes, 1):
            model_fixes_section += f"\nFix {i}: {fix['description']}\n"
            model_fixes_section += f"Problem:\n```python\n{fix['problem_code']}\n```\n"
            model_fixes_section += f"Fix:\n```python\n{fix['fixed_code']}\n```\n"
            model_fixes_section += f"Reason: {fix['reason']}\n"
        print(f"🔧 Found {len(model_fixes)} model-specific fixes to apply")

    # Call Groq LLM
    print("🤖 Calling LLM to generate model training code...")

    messages = [
        SystemMessage(content=MODELER_SYSTEM_PROMPT),
        HumanMessage(content=MODELER_USER_PROMPT.format(
            user_goal=user_goal,
            feature_result=feature_result[:2000],
            cleaning_code=cleaning_code[:1500],
            feature_code=feature_code[:1500],
            imbalance_ratio=imbalance_ratio,
            model_fixes_section=model_fixes_section,
            target_column=target_col,
        )),
    ]

    response, model_used = call_llm_with_fallback(messages, temperature=0.1, max_tokens=4096)
    print(f"  📡 Model used: {model_used}")
    model_code = extract_code_block(response.content)

    if not model_code:
        print("⚠️ No code block found, using full response")
        model_code = response.content

    save_code_to_file(model_code, "04_model_training_code.py")
    print("✅ Model training code generated")

    # Execute in shared E2B sandbox with retry
    # featured_data.csv already exists in sandbox from feature_eng agent
    print("⚡ Executing model training code in shared sandbox...")
    execution_result = ""
    success = False
    max_retries = 2

    executor = get_shared_sandbox()

    for attempt in range(max_retries + 1):
        result = executor.execute_code(model_code, timeout=180)  # longer timeout for training

        if result["success"]:
            execution_result = result["stdout"]
            success = True
            print(f"✅ Model training executed successfully (attempt {attempt + 1})")
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
                fix_response, _ = call_llm_with_fallback(fix_messages, temperature=0.1, max_tokens=4096)
                model_code = extract_code_block(fix_response.content)
                if not model_code:
                    model_code = fix_response.content
                save_code_to_file(model_code, "04_model_training_code.py")
            else:
                execution_result = f"FAILED after {max_retries + 1} attempts. Last error: {error_msg}"

    # Save report
    report = f"# Model Training Report\n\n"
    report += f"## Status: {'SUCCESS' if success else 'FAILED'}\n\n"
    report += f"## Generated Code\n```python\n{model_code}\n```\n\n"
    report += f"## Training Results\n```\n{execution_result}\n```\n"
    save_report(report, "04_model_training_report.md")

    print(f"\n📋 Model Training Result Preview:")
    print(execution_result[:800] if execution_result else "No output")

    # After successful execution, try to get visualization data
    viz_data = {}
    if success:
        try:
            print("📊 Retrieving visualization data...")
            viz_script = """
import json
try:
    with open('/home/user/visualization_data.json', 'r') as f:
        print(json.dumps(json.load(f)))
except Exception as e:
    print(f"Error reading viz data: {e}")
"""
            viz_result = executor.execute_code(viz_script)
            if viz_result["success"] and viz_result["stdout"]:
                import json
                try:
                    # Clean output to find JSON if there's other text
                    output = viz_result["stdout"].strip()
                    # Find the last line that looks like JSON if there's noise
                    lines = output.split('\n')
                    for line in reversed(lines):
                        if line.startswith('{') and line.endswith('}'):
                            viz_data = json.loads(line)
                            print("📊 Visualization data captured!")
                            break
                    if not viz_data and output.startswith('{'):
                         viz_data = json.loads(output)
                         print("📊 Visualization data captured!")
                except json.JSONDecodeError:
                    print(f"⚠️ Could not parse visualization data JSON: {viz_result['stdout'][:100]}...")
            else:
                 print(f"⚠️ No visualization data output found. Stdout: {viz_result.get('stdout', '')}")

        except Exception as e:
            print(f"⚠️ Could not load visualization data: {e}")

    return {
        "current_agent": "modeler",
        "model_code": model_code,
        "model_approved": True,
        "model_result": execution_result,
        "visualization_data": viz_data,
    }


def extract_code_block(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    # Fallback to general block
    if not match:
        pattern = r"```\s*\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""
