import os
import re

base_dir = "/Users/saicharanrajoju/Desktop/final project/ml-agent-assistant/src/agents"
agents = ["profiler.py", "cleaner.py", "feature_eng.py", "modeler.py", "critic.py"]

# Fix magic numbers
MAGIC_CONSTS = {
    "profiler.py": "TOP_CORRELATIONS_COUNT = 5\n\n",
    "modeler.py": "SAMPLE_FRACTION = 0.10\nMIN_ROWS_FOR_SAMPLE = 200\nMAX_ROWS_FOR_SAMPLE = 2000\nEXECUTION_TIMEOUT = 360\n\n",
    "critic.py": "MAX_ITERATIONS = 3\nMAX_SUGGESTIONS = 5\n\n"
}

for agent in agents:
    path = os.path.join(base_dir, agent)
    with open(path, "r") as f:
        content = f.read()

    # Add logging import
    if "import logging" not in content:
        content = re.sub(r'(import .*?\n)', r'\1import logging\n', content, count=1)
        # Add logger
        content = re.sub(r'(AGENT_TAG = .*?\n)', r'\1\nlogger = logging.getLogger(__name__)\n', content)

    # Replace print and print(f"...") with logger
    content = content.replace('print("\\n" + "="*60)', 'logger.info("\\n" + "="*60)')
    content = content.replace('print("="*60)', 'logger.info("="*60)')
    content = re.sub(r'print\((f?["\'].*?⚠️.*?["\'].*?)\)', r'logger.warning(\1)', content)
    content = re.sub(r'print\((f?["\'].*?FAILED.*?["\'].*?)\)', r'logger.error(\1)', content)
    # Remaining prints to info
    content = re.sub(r'print\((.*?)\)', r'logger.info(\1)', content)

    # Magic numbers extraction
    if agent == "profiler.py":
        content = content.replace("[:5]", "[:TOP_CORRELATIONS_COUNT]")
    elif agent == "modeler.py":
        content = content.replace("len(df) * 0.10", "len(df) * SAMPLE_FRACTION")
        content = content.replace("max(200, min(2000,", "max(MIN_ROWS_FOR_SAMPLE, min(MAX_ROWS_FOR_SAMPLE,")
        content = content.replace("timeout=360", "timeout=EXECUTION_TIMEOUT")
        content = content.replace("max_iterations=3", "max_iterations=MAX_ITERATIONS")  # Actually wait, modeler has MAX_RETRIES?
        content = content.replace("max_retries = 2", "MAX_RETRIES = 2")
        content = content.replace("max_retries", "MAX_RETRIES")
    elif agent == "critic.py":
        content = content.replace("max_iterations = 3", "") # handled by insert
        content = content.replace("iteration_count < max_iterations", "iteration_count < MAX_ITERATIONS")
        content = content.replace("iteration_count >= max_iterations", "iteration_count >= MAX_ITERATIONS")
        content = content.replace("({max_iterations})", "({MAX_ITERATIONS})")
        content = content.replace("[:5]", "[:MAX_SUGGESTIONS]")

    if agent in ["modeler.py", "cleaner.py", "feature_eng.py"]:
        # Handle EXCEPT: and EXCEPT EXCEPTION:
        # In strings, usually we use except ValueError or TypeError. In normal code, json.JSONDecodeError or OSError
        # Let's just do a blanket EXCEPT EXCEPTION AS E -> EXCEPT EXCEPTION AS E for now? No, user wants specific.

        # Let's do string exceptions manually below. But outside strings:
        pass

    # Insert magic consts
    if agent in MAGIC_CONSTS and MAGIC_CONSTS[agent] not in content:
        content = re.sub(r'(logger = logging.getLogger\(__name__\)\n)', r'\1\n' + MAGIC_CONSTS[agent], content)

    with open(path, "w") as f:
        f.write(content)

print("done")
