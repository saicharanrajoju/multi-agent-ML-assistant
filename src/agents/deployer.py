import re
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.tools.code_executor import get_shared_sandbox
from src.tools.file_utils import save_code_to_file, save_report
from src.prompts.deployer_prompt import DEPLOYER_SYSTEM_PROMPT, DEPLOYER_USER_PROMPT
from src.llm_helper import call_llm_with_fallback

def deployer_node(state: AgentState) -> dict:
    """
    Deployer Agent: Generates a complete FastAPI + Docker deployment package.
    
    Steps:
    1. Get model results and pipeline code from state
    2. Call Groq LLM to generate deployment package creator script
    3. Re-run cleaning, feature eng, and model training in sandbox to have model files
    4. Execute the deployment generator script
    5. Download the deployment package locally
    6. Return deployment code and results
    """
    print("\n" + "="*60)
    print("🚀 DEPLOYER AGENT")
    print("="*60)

    user_goal = state["user_goal"]
    model_result = state.get("model_result", "No model results available")
    cleaning_code = state.get("cleaning_code", "")
    feature_code = state.get("feature_code", "")
    target_col = state.get("target_column", "Churn")

    # Call Groq LLM
    print("🤖 Calling LLM to generate deployment package...")

    messages = [
        SystemMessage(content=DEPLOYER_SYSTEM_PROMPT),
        HumanMessage(content=DEPLOYER_USER_PROMPT.format(
            user_goal=user_goal,
            model_result=model_result[:2000],
            cleaning_code=cleaning_code[:1500],
            feature_code=feature_code[:1500],
            target_column=target_col,
        )),
    ]

    response, model_used = call_llm_with_fallback(messages, temperature=0.1, max_tokens=8000)
    print(f"  📡 Model used: {model_used}")
    deployment_code = extract_code_block(response.content)

    if not deployment_code:
        print("⚠️ No code block found, using full response")
        deployment_code = response.content

    save_code_to_file(deployment_code, "06_deployment_generator.py")
    print("✅ Deployment code generated")

    # Execute in shared E2B sandbox
    # All model files already exist in sandbox from previous agents
    print("⚡ Building deployment package in shared sandbox...")
    execution_result = ""
    success = False
    max_retries = 2

    executor = get_shared_sandbox()

    for attempt in range(max_retries + 1):
        result = executor.execute_code(deployment_code, timeout=120)

        if result["success"]:
            execution_result = result["stdout"]
            success = True
            print(f"✅ Deployment package generated (attempt {attempt + 1})")
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
                fix_response, _ = call_llm_with_fallback(fix_messages, temperature=0.1, max_tokens=8000)
                deployment_code = extract_code_block(fix_response.content)
                if not deployment_code:
                    deployment_code = fix_response.content
                save_code_to_file(deployment_code, "06_deployment_generator.py")
            else:
                execution_result = f"FAILED after {max_retries + 1} attempts. Last error: {error_msg}"

    # Try to download deployment files
    if success:
        try:
            import os
            deploy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs", "deployment")
            os.makedirs(deploy_dir, exist_ok=True)

            for filename in ["app.py", "requirements.txt", "Dockerfile", "docker-compose.yml", "test_api.py", "README.md"]:
                try:
                    executor.download_file(f"/home/user/deployment/{filename}", os.path.join(deploy_dir, filename))
                except Exception as e:
                    print(f"⚠️ Could not download {filename}: {e}")

            # Try to download model files
            for model_file in ["best_model.joblib", "preprocessor.joblib"]:
                try:
                    executor.download_file(f"/home/user/deployment/{model_file}", os.path.join(deploy_dir, model_file))
                except Exception as e:
                    print(f"⚠️ Could not download {model_file}: {e}")

            print(f"📁 Deployment files saved to outputs/deployment/")
        except Exception as e:
            print(f"⚠️ Could not download deployment files: {e}")

    # Save report
    report = f"# Deployment Report\n\n"
    report += f"## Status: {'SUCCESS' if success else 'FAILED'}\n\n"
    report += f"## Generated Deployment Script\n```python\n{deployment_code}\n```\n\n"
    report += f"## Execution Output\n```\n{execution_result}\n```\n\n"
    report += f"## Deployment Instructions\n"
    report += f"1. Navigate to outputs/deployment/\n"
    report += f"2. Run: docker-compose up --build\n"
    report += f"3. Test: python test_api.py\n"
    report += f"4. API available at: http://localhost:8000\n"
    save_report(report, "06_deployment_report.md")

    print(f"\n📋 Deployment Result Preview:")
    print(execution_result[:500] if execution_result else "No output")

    return {
        "current_agent": "deployer",
        "deployment_code": deployment_code,
        "deployment_approved": True,
        "api_endpoint": "http://localhost:8000",
    }


def extract_code_block(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    # Fallback if no specific language tag
    if not match:
        pattern = r"```\s*\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""
