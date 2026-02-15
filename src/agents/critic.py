import re
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.tools.file_utils import save_report
from src.prompts.critic_prompt import CRITIC_SYSTEM_PROMPT, CRITIC_USER_PROMPT
from src.llm_helper import call_llm_with_fallback

def extract_section(text: str, start_header: str, end_header: str) -> str:
    """Extract content between two ## headers."""
    pattern = rf"## {start_header}\s*\n(.*?)(?=## {end_header}|$)"
    match = re.search(pattern, text, re.DOTALL)
    # Fallback to search until next ## if end header not found immediately
    if not match:
        pattern = rf"## {start_header}\s*\n(.*?)(?=##|$)"
        match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text

def extract_severity(text: str) -> str:
    """Extract severity level."""
    pattern = r"## SEVERITY\s*\n\s*(CRITICAL|MODERATE|MINOR)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback: search anywhere in text
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
    
    # Find the CODE FIXES section
    pattern = r"## CODE FIXES\s*\n(.*?)(?=## [A-Z]|$)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return fixes
    
    section = match.group(1)
    
    # Check if no fixes needed
    if "no code fixes needed" in section.lower() or "pipeline looks good" in section.lower():
        return fixes
    
    # Parse individual fixes
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
    
    This agent does NOT execute any code — it only analyzes.
    
    Steps:
    1. Gather all code and results from state
    2. Send everything to Groq LLM with the critic prompt
    3. Parse the response to extract critique, severity, suggestions
    4. Determine if another iteration is needed
    5. Return critique report and routing decision
    """
    print("\n" + "="*60)
    print("🧐 CRITIC AGENT")
    print("="*60)

    iteration_count = state.get("iteration_count", 0) + 1
    user_goal = state["user_goal"]

    print(f"📝 Review iteration: {iteration_count}")

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
    # Try to guess n_features from columns_after if available, or just say "See below"
    n_features = len(cleaning_summary.get("columns_after", [])) if cleaning_summary.get("columns_after") else "Unknown"

    # Call Groq LLM
    print("🤖 Calling LLM for pipeline critique...")
    feature_count = len(dataset_summary.get("columns", []))
    target_col = state.get("target_column", "Churn")

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
            profile_report=profile_report[:1500],
            cleaning_code=cleaning_code[:1500],
            cleaning_result=cleaning_result[:1000],
            feature_code=feature_code[:1500],
            feature_result=feature_result[:1000],
            model_code=model_code[:2000],
            model_result=model_result[:1500],
        )),
    ]

    response, model_used = call_llm_with_fallback(messages, temperature=0.2, max_tokens=4096)
    print(f"  📡 Model used: {model_used}")
    critic_output = response.content
    print("✅ Critique complete")

    # Parse the response
    critique_report = extract_section(critic_output, "CRITIQUE REPORT", "SEVERITY")
    severity = extract_severity(critic_output)
    should_iterate_raw = extract_should_iterate(critic_output)
    improvement_suggestions = extract_suggestions(critic_output)
    scorecard = extract_scorecard(critic_output)
    code_fixes = extract_code_fixes(critic_output)

    # Determine if we should iterate
    # Only iterate if: critic says YES, severity is not MINOR, and we haven't exceeded max iterations
    max_iterations = 3
    should_iterate = (
        should_iterate_raw
        and severity in ["CRITICAL", "MODERATE"]
        and iteration_count < max_iterations
    )

    if should_iterate:
        print(f"🔄 Critic recommends iteration — severity: {severity}")
        print(f"   Suggestions: {len(improvement_suggestions)}")
        print(f"   Code Fixes: {len(code_fixes)}")
    else:
        if iteration_count >= max_iterations:
            print(f"⏹️ Max iterations ({max_iterations}) reached — proceeding to deployment")
        else:
            print(f"✅ Critic satisfied — severity: {severity} — proceeding to deployment")

    # Print summary
    print(f"\n📋 Critique Summary:")
    print(f"   Severity: {severity}")
    print(f"   Should iterate: {should_iterate}")
    print(f"   Suggestions ({len(improvement_suggestions)}):")
    for i, suggestion in enumerate(improvement_suggestions[:5], 1):
        print(f"     {i}. {suggestion[:100]}")
    
    if code_fixes:
        print(f"   Code Fixes ({len(code_fixes)}):")
        for i, fix in enumerate(code_fixes, 1):
            print(f"     {i}. {fix['description'][:100]} (in {fix['file']})")

    if scorecard:
        print("\n📊 SCORECARD:")
        for k, v in scorecard.items():
            print(f"   {k.replace('_', ' ').title()}: {v}/10")

    # Save report
    report = f"# Critique Report — Iteration {iteration_count}\n\n"
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
