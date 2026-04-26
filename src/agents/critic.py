import re
import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.tools.file_utils import save_report, extract_section
from src.prompts.critic_prompt import CRITIC_SYSTEM_PROMPT, CRITIC_USER_PROMPT
from src.llm_helper import call_llm_with_fallback

AGENT_TAG = "[Critic]"

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3
MAX_SUGGESTIONS = 5



def extract_severity(text: str) -> str:
    """Extract severity level."""
    pattern = r"## SEVERITY\s*\n\s*(CRITICAL|MODERATE|MINOR)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    for level in ["CRITICAL", "MODERATE", "MINOR"]:
        if level in text.upper():
            return level
    return "MINOR"

def extract_should_iterate(text: str) -> bool:
    """Extract whether the critic wants another iteration."""
    pattern = r"## SHOULD ITERATE\s*\n\s*(YES|NO)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).upper() == "YES"
    return False

def extract_suggestions(text: str) -> list:
    """Extract improvement suggestions."""
    pattern = r"## IMPROVEMENT SUGGESTIONS\s*\n(.*?)(?=##|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        section = match.group(1)
        suggestions = [line.strip().lstrip("- ").strip() for line in section.split("\n") if line.strip().startswith("-")]
        return suggestions if suggestions else ["No specific suggestions"]
    return ["No specific suggestions"]

def extract_scorecard(text: str) -> dict:
    """Extract scorecard scores from critic output."""
    scorecard = {}
    patterns = {
        "data_leakage": r"Data Leakage Prevention:\s*(\d+)/10",
        "code_quality": r"Code Quality:\s*(\d+)/10",
        "metric_alignment": r"Metric Alignment:\s*(\d+)/10",
        "feature_engineering": r"Feature Engineering:\s*(\d+)/10",
        "model_selection": r"Model Selection:\s*(\d+)/10",
        "deployment_readiness": r"Deployment Readiness:\s*(\d+)/10",
        "overall": r"OVERALL:\s*([\d.]+)/10",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            scorecard[key] = float(match.group(1))

    if "overall" not in scorecard and scorecard:
        scorecard["overall"] = round(sum(v for v in scorecard.values()) / len(scorecard), 1)

    return scorecard


def extract_code_fixes(text: str) -> list[dict]:
    """Extract specific code fixes from critic output."""
    fixes = []

    pattern = r"## CODE FIXES\s*\n(.*?)(?=## [A-Z]|$)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return fixes

    section = match.group(1)

    if "no code fixes needed" in section.lower() or "pipeline looks good" in section.lower():
        return fixes

    fix_pattern = r"### Fix \d+:\s*(.*?)\n\*\*File:\*\*\s*(.*?)\n\*\*Problem line:\*\*\s*```python\s*\n(.*?)```\s*\n\*\*Fixed code:\*\*\s*```python\s*\n(.*?)```\s*\n\*\*Why:\*\*\s*(.*?)(?=### Fix|\Z)"

    for match in re.finditer(fix_pattern, section, re.DOTALL):
        fixes.append({
            "description": match.group(1).strip(),
            "file": match.group(2).strip(),
            "problem_code": match.group(3).strip(),
            "fixed_code": match.group(4).strip(),
            "reason": match.group(5).strip(),
        })

    return fixes


def critic_node(state: AgentState) -> dict:
    """
    Critic Agent: Reviews the entire ML pipeline for errors, leakage, and quality issues.
    This agent does NOT execute any code -- it only analyzes.
    """
    logger.info("\n" + "="*60)
    logger.info(f"🧐 {AGENT_TAG} CRITIC AGENT")
    logger.info("="*60)

    iteration_count = state.get("iteration_count", 0) + 1
    
    reasoning_context = state.get("reasoning_context", {})
    problem_type = state.get("problem_type", "binary_classification")
    recommended_metric = state.get("recommended_metric", "f1")
    imbalance_strategy = reasoning_context.get("imbalance_strategy", "none")
    recommended_models = reasoning_context.get("recommended_models", [])

    user_goal = state["user_goal"]

    logger.info(f"📝 {AGENT_TAG} Review iteration: {iteration_count}")

    # Gather all pipeline artifacts
    profile_report = state.get("profile_report", "Not available")
    cleaning_code = state.get("cleaning_code", "Not available")
    cleaning_result = state.get("cleaning_result", "Not available")
    feature_code = state.get("feature_code", "Not available")
    feature_result = state.get("feature_result", "Not available")
    model_code = state.get("model_code", "Not available")
    model_result = state.get("model_result", "Not available")

    # Extract context stats
    dataset_summary = state.get("dataset_summary", {})
    cleaning_summary = state.get("cleaning_summary", {})

    dataset_shape = dataset_summary.get("shape", "Unknown")
    imbalance_ratio = dataset_summary.get("class_imbalance_ratio", "Unknown")
    n_features = len(cleaning_summary.get("columns_after", [])) if cleaning_summary.get("columns_after") else "Unknown"

    # Call Groq LLM
    logger.info(f"🤖 {AGENT_TAG} Calling LLM for pipeline critique...")
    feature_count = len(dataset_summary.get("columns", []))
    target_col = state.get("target_column", "")

    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=CRITIC_USER_PROMPT.format(
            user_goal=user_goal,
            iteration_count=iteration_count,
            dataset_shape=dataset_shape,
            imbalance_ratio=imbalance_ratio,
            n_features=n_features,
            feature_count=feature_count,
            target_column=target_col,
            problem_type=problem_type,
            recommended_metric=recommended_metric,
            imbalance_strategy=imbalance_strategy,
            recommended_models=recommended_models,
            profile_report=profile_report[:1500],
            cleaning_code=cleaning_code,
            cleaning_result=cleaning_result[:1000],
            feature_code=feature_code,
            feature_result=feature_result[:1000],
            model_code=model_code,
            model_result=model_result[:1500],
        )),
    ]

    response, model_used = call_llm_with_fallback(messages, temperature=0.2)
    logger.info(f"  📡 {AGENT_TAG} Model used: {model_used}")
    critic_output = response.content
    logger.info(f"✅ {AGENT_TAG} Critique complete")

    # Parse the response
    critique_report = extract_section(critic_output, "CRITIQUE REPORT", "METADATA")
    
    metadata = {}
    json_match = re.search(r'```json\s*(.*?)\s*```', critic_output, re.DOTALL | re.IGNORECASE)
    if json_match:
        try:
            metadata = json.loads(json_match.group(1))
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ {AGENT_TAG} Warning: Failed to parse JSON metadata block: {e}")

    if "severity" in metadata:
        severity = str(metadata["severity"]).upper()
    else:
        logger.warning(f"⚠️ {AGENT_TAG} Warning: Falling back to regex for severity")
        severity = extract_severity(critic_output)

    if "should_iterate" in metadata:
        should_iterate_raw = bool(metadata["should_iterate"])
    else:
        logger.warning(f"⚠️ {AGENT_TAG} Warning: Falling back to regex for should_iterate")
        should_iterate_raw = extract_should_iterate(critic_output)

    if "suggestions" in metadata:
        improvement_suggestions = metadata["suggestions"]
    else:
        logger.warning(f"⚠️ {AGENT_TAG} Warning: Falling back to regex for suggestions")
        improvement_suggestions = extract_suggestions(critic_output)

    if "scorecard" in metadata:
        scorecard = metadata["scorecard"]
    else:
        logger.warning(f"⚠️ {AGENT_TAG} Warning: Falling back to regex for scorecard")
        scorecard = extract_scorecard(critic_output)

    if "code_fixes" in metadata:
        code_fixes = metadata["code_fixes"]
    else:
        logger.warning(f"⚠️ {AGENT_TAG} Warning: Falling back to regex for code_fixes")
        code_fixes = extract_code_fixes(critic_output)

    # Determine if we should iterate
    
    should_iterate = (
        should_iterate_raw
        and severity in ["CRITICAL", "MODERATE"]
        and iteration_count < MAX_ITERATIONS
    )

    if should_iterate:
        logger.info(f"🔄 {AGENT_TAG} Recommends iteration — severity: {severity}")
        logger.info(f"   Suggestions: {len(improvement_suggestions)}")
        logger.info(f"   Code Fixes: {len(code_fixes)}")
    else:
        if iteration_count >= MAX_ITERATIONS:
            logger.info(f"⏹️ {AGENT_TAG} Max iterations ({MAX_ITERATIONS}) reached — proceeding to deployment")
        else:
            logger.info(f"✅ {AGENT_TAG} Satisfied — severity: {severity} — proceeding to deployment")

    # Print summary
    logger.info(f"\n📋 {AGENT_TAG} Critique Summary:")
    logger.info(f"   Severity: {severity}")
    logger.info(f"   Should iterate: {should_iterate}")
    logger.info(f"   Suggestions ({len(improvement_suggestions)}):")
    for i, suggestion in enumerate(improvement_suggestions[:MAX_SUGGESTIONS], 1):
        logger.info(f"     {i}. {suggestion[:100]}")

    if code_fixes:
        logger.info(f"   Code Fixes ({len(code_fixes)}):")
        for i, fix in enumerate(code_fixes, 1):
            logger.info(f"     {i}. {fix['description'][:100]} (in {fix['file']})")

    if scorecard:
        logger.info(f"\n📊 {AGENT_TAG} SCORECARD:")
        for k, v in scorecard.items():
            logger.info(f"   {k.replace('_', ' ').title()}: {v}/10")

    # Save report
    report = f"# Critique Report -- Iteration {iteration_count}\n\n"
    report += f"## Severity: {severity}\n\n"
    report += f"## Should Iterate: {'YES' if should_iterate else 'NO'}\n\n"
    report += f"## Detailed Critique\n{critique_report}\n\n"
    report += f"## Code Fixes\n"
    if code_fixes:
        for i, fix in enumerate(code_fixes, 1):
            report += f"### Fix {i}: {fix['description']}\n"
            report += f"**File:** {fix['file']}\n"
            report += f"**Reason:** {fix['reason']}\n\n"
            report += f"```python\n# Problem:\n{fix['problem_code']}\n# Fix:\n{fix['fixed_code']}\n```\n\n"
    else:
        report += "No specific code fixes provided.\n\n"

    report += f"## Improvement Suggestions\n"
    for s in improvement_suggestions:
        report += f"- {s}\n"
    save_report(report, f"05_critique_report_iter{iteration_count}.md")

    # Update iteration history
    iteration_entry = {
        "iteration": iteration_count,
        "severity": severity,
        "n_fixes": len(code_fixes),
        "suggestions": improvement_suggestions,
        "scorecard": scorecard if scorecard else {},
        "viz_snapshot": state.get("visualization_data", {}),
    }

    existing_history = state.get("iteration_history", [])
    new_history = existing_history + [iteration_entry]

    return {
        "current_agent": "critic",
        "critique_report": critique_report,
        "improvement_suggestions": improvement_suggestions,
        "code_fixes": code_fixes,
        "iteration_history": new_history,
        "iteration_count": iteration_count,
        "should_iterate": should_iterate,
        "scorecard": scorecard,
    }
