import os
import logging
import json
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from langchain_core.runnables import RunnableConfig
from src.tools.code_executor import get_sandbox_for_run
from src.tools.file_utils import save_code_to_file, save_report, extract_code_block, build_fix_prompt
from src.prompts.deployer_prompt import DEPLOYER_SYSTEM_PROMPT, DEPLOYER_USER_PROMPT
from src.llm_helper import call_llm_with_fallback

AGENT_TAG = "[Deployer]"

logger = logging.getLogger(__name__)

def run_hardware_profile(executor, problem_type: str, target_col: str) -> dict:
    """
    Profiles the saved model's inference latency and memory footprint inside E2B.
    Returns a dict with latency_ms_per_1k, memory_mb, throughput_rows_per_sec.
    """
    profile_code = """
import os, json, time, tracemalloc, warnings
import numpy as np
warnings.filterwarnings('ignore')

results = {}
try:
    import joblib
    import pandas as pd


    # Find model artifact
    model_paths = [
        '/home/user/deployment/best_model.joblib',
        '/home/user/best_model.joblib',
        '/home/user/best_model.pkl',
    ]
    preprocessor_paths = [
        '/home/user/deployment/preprocessor.joblib',
        '/home/user/preprocessor.joblib',
        '/home/user/preprocessor.pkl',
        '/home/user/preprocessing_pipeline.pkl',
    ]

    model_path = next((p for p in model_paths if os.path.exists(p)), None)
    pre_path = next((p for p in preprocessor_paths if os.path.exists(p)), None)

    if not model_path:
        results = {'error': 'No model artifact found', 'profiled': False}
        print(json.dumps(results))
        exit()

    model = joblib.load(model_path)
    preprocessor = joblib.load(pre_path) if pre_path else None

    # Load a sample for profiling
    df = pd.read_csv('/home/user/featured_data.csv').drop(columns=['TARGET'], errors='ignore')
    df = df.fillna(df.median(numeric_only=True))
    sample = df.head(1000)

    # Preprocess if preprocessor exists
    if preprocessor is not None:
        try:
            sample_processed = preprocessor.transform(sample)
        except (ValueError, TypeError, KeyError):
            sample_processed = sample.values
    else:
        sample_processed = sample.values

    # Warm up
    try:
        _ = model.predict(sample_processed[:10])
    except ValueError:
        results = {'error': 'Model.predict() failed on sample', 'profiled': False}
        print(json.dumps(results))
        exit()

    # Latency benchmark: 10 runs of predict on 1000 rows
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        model.predict(sample_processed)
        times.append((time.perf_counter() - t0) * 1000)  # ms

    median_latency_ms = round(float(np.median(times)), 2)
    throughput = round(1000 / (median_latency_ms / 1000), 0)  # rows per second

    # Memory benchmark
    tracemalloc.start()
    model.predict(sample_processed)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = round(peak / 1024 / 1024, 2)

    # Model file size
    model_size_mb = round(os.path.getsize(model_path) / 1024 / 1024, 3)

    results = {
        'profiled': True,
        'latency_ms_per_1k_rows': median_latency_ms,
        'peak_memory_mb': peak_mb,
        'model_file_size_mb': model_size_mb,
        'throughput_rows_per_sec': int(throughput),
        'sample_size': len(sample),
        'verdict': 'FAST' if median_latency_ms < 100 else 'MODERATE' if median_latency_ms < 500 else 'SLOW',
    }
except RuntimeError as e:
    results = {'error': str(e)[:200], 'profiled': False}

print(json.dumps(results))
"""
    profile_code = profile_code.replace('TARGET', target_col)
    res = executor.execute_code(profile_code, timeout=60)
    if res["success"]:
        for line in reversed(res["stdout"].strip().split('\n')):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    return json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ {AGENT_TAG} Warning: Skipping malformed JSON line: {str(e)}")
                    continue
    return {"profiled": False, "error": "Execution failed"}

def deployer_node(state: AgentState, config: RunnableConfig = None) -> dict:
    """
    Deployer Agent: Generates a Streamlit prediction app deployment package.
    """
    logger.info("\n" + "="*60)
    logger.info(f"🚀 {AGENT_TAG} DEPLOYER AGENT")
    logger.info("="*60)

    user_goal = state["user_goal"]
    model_result = state.get("model_result", "No model results available")
    cleaning_code = state.get("cleaning_code", "")
    feature_code = state.get("feature_code", "")
    target_col = state.get("target_column", "")
    problem_type = state.get("problem_type", "binary_classification")

    # Call Groq LLM
    logger.info(f"🤖 {AGENT_TAG} Calling LLM to generate deployment package...")

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

    response, model_used = call_llm_with_fallback(messages, temperature=0.1)
    logger.info(f"  📡 {AGENT_TAG} Model used: {model_used}")
    deployment_code = extract_code_block(response.content)

    if not deployment_code:
        logger.warning(f"⚠️ {AGENT_TAG} Warning: No code block found, using full response")
        deployment_code = response.content

    save_code_to_file(deployment_code, "06_deployment_generator.py")
    logger.info(f"✅ {AGENT_TAG} Deployment code generated")

    # Execute in shared E2B sandbox
    logger.info(f"⚡ {AGENT_TAG} Building deployment package in shared sandbox...")
    execution_result = ""
    success = False
    max_retries = 2

    run_id = (config or {}).get("configurable", {}).get("thread_id", "default")
    executor = get_sandbox_for_run(run_id)

    for attempt in range(max_retries + 1):
        result = executor.execute_code(deployment_code, timeout=120)

        if result["success"]:
            execution_result = result["stdout"]
            success = True
            logger.info(f"✅ {AGENT_TAG} Deployment package generated (attempt {attempt + 1})")
            break
        else:
            error_msg = result["error"]
            logger.warning(f"⚠️ {AGENT_TAG} Warning: Attempt {attempt + 1} failed: {error_msg[:200]}")

            if attempt < max_retries:
                logger.info(f"🔧 {AGENT_TAG} Asking LLM to fix the code (retry {attempt + 1})...")
                fix_messages = messages + [
                    HumanMessage(content=build_fix_prompt(deployment_code, error_msg, attempt))
                ]
                fix_response, _ = call_llm_with_fallback(fix_messages, temperature=min(0.1 + attempt * 0.15, 0.5))
                deployment_code = extract_code_block(fix_response.content) or fix_response.content
                save_code_to_file(deployment_code, "06_deployment_generator.py")
            else:
                execution_result = f"FAILED after {max_retries + 1} attempts. Last error: {error_msg}"

    # Try to download deployment files
    if success:
        try:
            deploy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs", "deployment")
            os.makedirs(deploy_dir, exist_ok=True)

            for filename in ["app.py", "requirements.txt"]:
                try:
                    executor.download_file(f"/home/user/deployment/{filename}", os.path.join(deploy_dir, filename))
                except Exception as e:
                    logger.warning(f"⚠️ {AGENT_TAG} Warning: Could not download {filename}: {str(e)[:200]}")

            for model_file in ["best_model.joblib", "preprocessor.joblib"]:
                try:
                    executor.download_file(f"/home/user/deployment/{model_file}", os.path.join(deploy_dir, model_file))
                except Exception as e:
                    logger.warning(f"⚠️ {AGENT_TAG} Warning: Could not download {model_file}: {str(e)[:200]}")

            logger.info(f"📁 {AGENT_TAG} Deployment files saved to outputs/deployment/")
        except Exception as e:
            logger.warning(f"⚠️ {AGENT_TAG} Warning: Could not download deployment files: {str(e)[:200]}")

        logger.info(f"⚡ {AGENT_TAG} Running hardware profiling...")
        hardware_profile = run_hardware_profile(executor, problem_type, target_col)

        if hardware_profile.get("profiled"):
            logger.info(f"📊 {AGENT_TAG} Hardware Profile:")
            logger.info(f"   Latency:    {hardware_profile['latency_ms_per_1k_rows']}ms per 1000 rows")
            logger.info(f"   Memory:     {hardware_profile['peak_memory_mb']}MB peak")
            logger.info(f"   Model size: {hardware_profile['model_file_size_mb']}MB")
            logger.info(f"   Throughput: {hardware_profile['throughput_rows_per_sec']} rows/sec")
            logger.info(f"   Verdict:    {hardware_profile['verdict']}")
        else:
            hardware_profile = {}
            logger.warning(f"⚠️ {AGENT_TAG} Hardware profiling skipped: {hardware_profile.get('error', 'unknown')}")
    else:
        hardware_profile = {}

    # Save report
    report = f"# Deployment Report\n\n"
    report += f"## Status: {'SUCCESS' if success else 'FAILED'}\n\n"
    report += f"## Generated Deployment Script\n```python\n{deployment_code}\n```\n\n"
    report += f"## Execution Output\n```\n{execution_result}\n```\n\n"
    report += f"## Deployment Instructions\n"
    report += f"1. Navigate to outputs/deployment/\n"
    report += f"2. Run: streamlit run app.py\n"
    report += f"3. Test: python test_api.py\n"
    report += f"4. API available at: http://localhost:8000\n"
    save_report(report, "06_deployment_report.md")

    logger.info(f"\n📋 {AGENT_TAG} Deployment Result Preview:")
    logger.info(execution_result[:500] if execution_result else "No output")

    return {
        "current_agent": "deployer",
        "deployment_code": deployment_code,
        "deployment_approved": success,
        "api_endpoint": "http://localhost:8000" if success else "",
        "hardware_profile": hardware_profile,
    }
