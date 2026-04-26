import time
from src.config import (
    GROQ_API_KEY, GROQ_API_KEYS, NVIDIA_API_KEY, NVIDIA_BASE_URL, GEMINI_API_KEY,
    TOGETHER_API_KEY, TOGETHER_BASE_URL, TOGETHER_MODELS,
    DEFAULT_MODEL, FALLBACK_MODEL, LLM_PROVIDER,
    GROQ_MODELS, NVIDIA_MODELS, GEMINI_MODELS,
)


def _build_llm(model_name: str, temperature: float, provider: str = None, api_key: str = None):
    """Instantiate the correct LangChain chat model based on provider."""
    p = provider or LLM_PROVIDER

    if p == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            google_api_key=GEMINI_API_KEY,
            model=model_name,
            temperature=temperature,
        )
    elif p == "nvidia":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=NVIDIA_API_KEY,
            base_url=NVIDIA_BASE_URL,
            model=model_name,
            temperature=temperature,
        )
    elif p == "together":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=api_key or TOGETHER_API_KEY,
            base_url=TOGETHER_BASE_URL,
            model=model_name,
            temperature=temperature,
        )
    else:  # groq
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=api_key or GROQ_API_KEY,
            model_name=model_name,
            temperature=temperature,
        )


def get_llm(temperature=0.1, model=None):
    """Get an LLM instance using the configured provider."""
    return _build_llm(model or DEFAULT_MODEL, temperature)


def call_llm_with_fallback(messages, temperature=0.1):
    """
    Call the LLM with automatic cross-provider fallback on rate limit errors.

    Cascade order:
      - If LLM_PROVIDER=gemini:  gemini-2.0-flash → gemini-1.5-flash → groq-70b → groq-8b
      - If LLM_PROVIDER=groq:    groq-70b → groq-8b
      - If LLM_PROVIDER=nvidia:  nvidia-70b → nvidia-8b → groq-8b

    Returns: (response, model_used_string)
    """

    def _is_rate_limit(error_str: str) -> bool:
        return any(x in error_str for x in ["429", "rate_limit", "rate limit", "RESOURCE_EXHAUSTED", "quota"])

    groq_primary_key = GROQ_API_KEYS[0] if GROQ_API_KEYS else None

    if LLM_PROVIDER == "together":
        cascade = [
            ("together", TOGETHER_MODELS["primary"],   None),
            ("together", TOGETHER_MODELS["fallback"],  None),
            ("together", TOGETHER_MODELS["fallback2"], None),
            ("together", TOGETHER_MODELS["fallback3"], None),
        ]
    elif LLM_PROVIDER == "gemini":
        cascade = [
            ("gemini", GEMINI_MODELS["primary"],  None),
            ("gemini", GEMINI_MODELS["fallback"],  None),
            ("groq",   GROQ_MODELS["primary"],     groq_primary_key),
            ("groq",   GROQ_MODELS["fallback"],    groq_primary_key),
        ]
    elif LLM_PROVIDER == "nvidia":
        cascade = [
            ("nvidia", NVIDIA_MODELS["primary"],  None),
            ("nvidia", NVIDIA_MODELS["fallback"],  None),
            ("groq",   GROQ_MODELS["fallback"],    groq_primary_key),
        ]
    else:  # groq — rotate through all available keys before falling back to 8b
        cascade = []
        for key in GROQ_API_KEYS:
            cascade.append(("groq", GROQ_MODELS["primary"], key))
        for key in GROQ_API_KEYS:
            cascade.append(("groq", GROQ_MODELS["fallback"], key))

    last_error = None
    for entry in cascade:
        provider, model_name = entry[0], entry[1]
        api_key = entry[2] if len(entry) > 2 else None
        label = f"{provider}/{model_name}"
        for attempt in range(2):
            try:
                llm = _build_llm(model_name, temperature, provider=provider, api_key=api_key)
                response = llm.invoke(messages)
                print(f"  ✅ LLM call successful [{label}]")
                return response, label

            except Exception as e:
                error_str = str(e)
                last_error = error_str

                if _is_rate_limit(error_str):
                    print(f"  ⚠️ Rate limited [{label}] — trying next in cascade...")
                    break
                else:
                    if attempt == 0:
                        wait = 5
                        print(f"  ⚠️ LLM error [{label}] attempt {attempt+1}: {error_str[:80]}. Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"  ❌ [{label}] failed twice, moving to next cascade option...")
                        break

    raise Exception(f"All models in cascade exhausted. Last error: {last_error}")
