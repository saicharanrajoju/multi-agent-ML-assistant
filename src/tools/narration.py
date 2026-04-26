import logging
from langchain_core.messages import SystemMessage, HumanMessage
from src.llm_helper import call_llm_with_fallback

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are an ML tutor explaining a machine learning pipeline step to a student who is new to data science. "
    "Write in plain, conversational English. Be specific — reference the actual numbers, column names, and "
    "decisions from the context. Explain the WHY behind each decision, not just what happened. "
    "Write 150-250 words as flowing prose (no bullet points). "
    "Do not start with 'I'. Do not use jargon without explaining it."
)


def _safe_str(v) -> str:
    """Convert a value to a readable string, capping long collections."""
    if isinstance(v, list):
        truncated = v[:10]
        suffix = f" (+{len(v) - 10} more)" if len(v) > 10 else ""
        return ", ".join(str(x) for x in truncated) + suffix
    if isinstance(v, dict):
        items = list(v.items())[:8]
        suffix = f" (+{len(v) - 8} more)" if len(v) > 8 else ""
        return ", ".join(f"{k}={val}" for k, val in items) + suffix
    return str(v)


def _build_context_str(context_dict: dict) -> str:
    lines = []
    for k, v in context_dict.items():
        if v is None or v == "" or v == [] or v == {}:
            continue
        lines.append(f"- {k}: {_safe_str(v)}")
    return "\n".join(lines)


def _build_prompt(agent_tag: str, context_dict: dict, context_str: str) -> str:
    if agent_tag == "profiler":
        return (
            f"The Data Profiler agent just finished analyzing the dataset. Key facts:\n\n"
            f"{context_str}\n\n"
            f"Explain to a student:\n"
            f"(1) Why this problem was classified as '{context_dict.get('problem_type', 'unknown')}' "
            f"and what that means for how the model will be trained.\n"
            f"(2) Why '{context_dict.get('recommended_metric', 'unknown')}' was chosen as the success metric "
            f"for this specific goal.\n"
            f"(3) Why the imbalance strategy '{context_dict.get('imbalance_strategy', 'none')}' was chosen "
            f"given the class imbalance ratio of {context_dict.get('imbalance_ratio', 'N/A')}.\n"
            f"(4) Why these specific models were recommended: {_safe_str(context_dict.get('recommended_models', []))}.\n"
            f"Be specific — reference column names, the actual numbers, and the data characteristics."
        )
    elif agent_tag == "cleaner":
        return (
            f"The Data Cleaner agent just finished cleaning the dataset. Key facts:\n\n"
            f"{context_str}\n\n"
            f"Explain to a student:\n"
            f"(1) What problems existed in the raw data and why each one needed to be fixed before training.\n"
            f"(2) Why each major cleaning decision was made — the encoding choices, imputation strategies, "
            f"and any columns that were dropped.\n"
            f"(3) What the cleaned data looks like now and why it is ready for machine learning.\n"
            f"Be specific — mention actual column names, missing value counts, and the reasoning behind each choice."
        )
    elif agent_tag == "feature_eng":
        return (
            f"The Feature Engineering agent just finished creating new features. Key facts:\n\n"
            f"{context_str}\n\n"
            f"Explain to a student:\n"
            f"(1) Why simply using the raw cleaned columns is not always enough — what new features add.\n"
            f"(2) Why specific transformations (log, interaction terms, binning) were applied and to which columns.\n"
            f"(3) Why some features might have been removed due to multicollinearity and why redundant features "
            f"hurt model performance.\n"
            f"Reference the actual column names and the mathematical reasoning behind each transformation."
        )
    elif agent_tag == "modeler":
        return (
            f"The Model Training agent just finished selecting and evaluating the best model. Key facts:\n\n"
            f"{context_str}\n\n"
            f"Explain to a student:\n"
            f"(1) Why the system first tested all models on just 10% of the data before full training — "
            f"what advantage does this 'model scout' give?\n"
            f"(2) Why '{context_dict.get('best_model', 'the winning model')}' won and what makes this model "
            f"type well-suited for this specific dataset and problem.\n"
            f"(3) What the performance numbers mean in plain English — is this a good result or not?\n"
            f"(4) Why cross-validation was used and what it proves about the model's reliability on unseen data.\n"
            f"Reference the actual model names, scores, dataset size, and imbalance handling."
        )
    return ""


def generate_narration(agent_tag: str, context_dict: dict) -> str:
    """
    Generate a plain-English tutor narration for a completed agent step.
    Returns empty string on any failure — never blocks the pipeline.

    Args:
        agent_tag: One of "profiler", "cleaner", "feature_eng", "modeler"
        context_dict: Key facts from the agent run (decisions, stats, column names)
    """
    if agent_tag not in ("profiler", "cleaner", "feature_eng", "modeler"):
        return ""

    try:
        context_str = _build_context_str(context_dict)
        prompt = _build_prompt(agent_tag, context_dict, context_str)
        if not prompt:
            return ""

        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ]

        response, _ = call_llm_with_fallback(messages, temperature=0.4)
        narration = response.content.strip()
        word_count = len(narration.split())
        logger.info(f"  📚 [{agent_tag}] Narration generated ({word_count} words)")
        return narration

    except Exception as e:
        logger.warning(
            f"  ⚠️ [{agent_tag}] Narration generation failed (non-blocking): {str(e)[:120]}"
        )
        return ""
