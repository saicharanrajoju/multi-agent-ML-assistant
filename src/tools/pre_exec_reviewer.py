"""
Pre-execution code reviewer.

Checks the LLM-generated inner block for a focused set of critical bugs
before it enters the execution loop. One extra LLM call per agent — catches
issues that cause silent wrong results or runtime crashes.
"""

from __future__ import annotations

import logging
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

_REVIEWER_SYSTEM_PROMPT = """You are a focused code reviewer for ML pipeline code. You check ONLY for five critical bug categories. You do not refactor, style-check, or suggest improvements — you only fix things that will cause wrong results or runtime crashes.

The five checks you perform:

1. PROBLEM_TYPE vs MODEL/METRIC ALIGNMENT (CRITICAL)
   - If problem_type contains 'regression': the code must use regressors (RandomForestRegressor,
     GradientBoostingRegressor, XGBRegressor, Ridge, Lasso) and metrics (RMSE, MAE, R²).
     If it uses any *Classifier class, accuracy_score, f1_score, roc_auc_score, or
     classification_report → that is a CRITICAL bug.
   - If problem_type contains 'classification': the code must use classifiers. If it uses any
     *Regressor class or mean_squared_error/r2_score as the primary evaluation → CRITICAL bug.

2. SMOTE APPLIED TO FULL DATA OR BEFORE SPLIT (CRITICAL)
   - SMOTE must only be applied inside a Pipeline (ImbPipeline), which ensures it only sees
     training data. If SMOTE is applied directly to X or df before the train/test split, or
     applied outside a pipeline to X_train after the split but before the pipeline → CRITICAL bug.

3. NUMPY SCALARS IN F-STRING FORMAT SPECS (RUNTIME TypeError)
   - Patterns like f"{some_numpy_value:.4f}" where some_numpy_value is a numpy scalar from
     .mean(), .std(), a corr() call, quantile(), etc. — these cause TypeError at runtime.
   - Fix: wrap with float(): f"{float(some_numpy_value):.4f}"
   - EXCEPTION: if the value was already cast to float earlier in the same code path, it is fine.

4. TARGET COLUMN LEAKAGE IN FEATURE CREATION
   - The target column must never be used as a source for creating new features.
   - If target_column appears in any arithmetic expression creating a new column → CRITICAL bug.

5. REQUIRED PRINT STATEMENTS PRESENT
   - The code must contain the required_prints strings listed in the context.
   - If any required print is missing, add it at the end of the code.

RESPONSE FORMAT — respond with exactly these two sections, nothing else:

ISSUES: ["description of issue 1", "description of issue 2"]  or  ISSUES: none

CORRECTED_CODE:
```python
<full corrected inner block — identical to input if no issues>
```
"""


def review_inner_block(
    agent_tag: str,
    inner_block: str,
    context: dict,
) -> tuple[str, list[str]]:
    """
    Pre-execution review of a generated inner block.

    Checks for: problem-type/model mismatch, SMOTE misuse, numpy f-string
    crashes, target leakage in features, and missing required prints.

    Returns (corrected_inner_block, issues_list).
    If no issues found, returns (original_inner_block, []).
    """
    from src.llm_helper import call_llm_with_fallback
    from src.tools.file_utils import extract_code_block

    problem_type = context.get("problem_type", "")
    target_column = context.get("target_column", "")
    agent_type = context.get("agent_type", "")
    required_prints = context.get("required_prints", [])

    required_prints_str = "\n".join(f"  - {p}" for p in required_prints) if required_prints else "  (none specified)"

    user_prompt = f"""Review this inner block for agent: {agent_type}

CONTEXT:
- problem_type: {problem_type}
- target_column: {target_column}
- required_prints (at least one of each must appear):
{required_prints_str}

INNER BLOCK TO REVIEW:
```python
{inner_block}
```

Check for ALL five issue categories in the system prompt. Be precise about line content.
If no issues are found, return ISSUES: none and CORRECTED_CODE with the original block unchanged.
"""

    try:
        messages = [
            SystemMessage(content=_REVIEWER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response, model_used = call_llm_with_fallback(messages, temperature=0.0)
        content = response.content

        # Parse ISSUES section
        issues: list[str] = []
        if "ISSUES:" in content:
            issues_line = content.split("ISSUES:")[1].split("\n")[0].strip()
            if issues_line.lower() not in ("none", "[]", ""):
                # Simple list parse — strip brackets and quotes, split on comma
                cleaned = issues_line.strip("[]")
                for item in cleaned.split('",'):
                    item = item.strip().strip('"').strip("'").strip(",").strip()
                    if item and item.lower() != "none":
                        issues.append(item)

        if not issues:
            return inner_block, []

        # Parse CORRECTED_CODE section
        corrected_code = inner_block
        if "CORRECTED_CODE:" in content:
            after_marker = content.split("CORRECTED_CODE:")[1]
            extracted = extract_code_block(after_marker)
            if extracted and extracted.strip():
                corrected_code = extracted.strip()
            elif after_marker.strip():
                # No code fence — take everything after the marker
                corrected_code = after_marker.strip()

        logger.info(f"{agent_tag} Pre-exec reviewer [{model_used}]: {len(issues)} issue(s) found and corrected")
        for issue in issues:
            logger.info(f"  CORRECTED: {issue}")

        return corrected_code, issues

    except Exception as exc:
        # Reviewer failure must never block execution — log and return original
        logger.warning(f"{agent_tag} Pre-exec reviewer failed (non-blocking): {str(exc)[:120]}")
        return inner_block, []
