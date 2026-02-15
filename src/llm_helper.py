import time
from langchain_groq import ChatGroq
from src.config import GROQ_API_KEY, DEFAULT_MODEL, FALLBACK_MODEL


def get_llm(temperature=0.1, max_tokens=4096, model=None):
    """Get an LLM instance. Tries the primary model first, falls back if rate limited."""
    target_model = model or DEFAULT_MODEL
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=target_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def call_llm_with_fallback(messages, temperature=0.1, max_tokens=4096):
    """
    Call the LLM with automatic fallback on rate limit errors.

    Tries the primary model (70B) first.
    If it gets a 429 rate limit error, automatically retries with the fallback model (8B).
    Also implements a simple retry with backoff for transient errors.

    Returns: (response, model_used)
    """
    models_to_try = [DEFAULT_MODEL, FALLBACK_MODEL]

    for model_name in models_to_try:
        for attempt in range(3):  # 3 retries per model
            try:
                llm = ChatGroq(
                    api_key=GROQ_API_KEY,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                response = llm.invoke(messages)
                print(f"  ✅ LLM call successful (model: {model_name})")
                return response, model_name
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate_limit" in error_str.lower():
                    if model_name == DEFAULT_MODEL:
                        print(f"  ⚠️ Rate limited on {model_name}, switching to fallback...")
                        break  # Break retry loop, try next model
                    else:
                        wait_time = 30 * (attempt + 1)
                        print(f"  ⚠️ Rate limited on fallback too. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                else:
                    if attempt < 2:
                        wait_time = 5 * (attempt + 1)
                        print(f"  ⚠️ LLM error (attempt {attempt+1}): {error_str[:100]}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise Exception(f"LLM failed after 3 attempts on {model_name}: {error_str}")

    raise Exception("All LLM models exhausted. Please wait for rate limits to reset.")
