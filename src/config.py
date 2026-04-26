import os
import warnings
from dotenv import load_dotenv

load_dotenv()

# Collect all Groq keys (GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, ...)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEYS = [k for k in [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
] if k]
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
E2B_API_KEY = os.getenv("E2B_API_KEY")

# "groq", "nvidia", "gemini", or "together"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

required_keys = [
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "E2B_API_KEY",
]

missing_keys = [key for key in required_keys if not os.getenv(key)]

if missing_keys:
    warnings.warn(
        f"Missing required environment variables: {', '.join(missing_keys)}. "
        f"Pipeline will fail at runtime if these are not set.",
        stacklevel=2,
    )


def validate_required_keys():
    """Call at runtime (not import time) to enforce required keys."""
    if missing_keys:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_keys)}")
    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        raise ValueError("LLM_PROVIDER=groq but GROQ_API_KEY is not set.")
    if LLM_PROVIDER == "nvidia" and not NVIDIA_API_KEY:
        raise ValueError("LLM_PROVIDER=nvidia but NVIDIA_API_KEY is not set.")
    if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        raise ValueError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set.")
    if LLM_PROVIDER == "together" and not TOGETHER_API_KEY:
        raise ValueError("LLM_PROVIDER=together but TOGETHER_API_KEY is not set.")


# Set LANGSMITH_TRACING to true explicitly
os.environ["LANGSMITH_TRACING"] = "true"

# --- Groq models ---
GROQ_MODELS = {
    "primary": "llama-3.3-70b-versatile",
    "fallback": "llama-3.1-8b-instant",
}

# --- NVIDIA NIM models (OpenAI-compatible endpoint) ---
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODELS = {
    "primary": "meta/llama-3.3-70b-instruct",
    "fallback": "meta/llama-3.1-8b-instruct",
}

# --- Google Gemini models ---
GEMINI_MODELS = {
    "primary": "gemini-2.0-flash",
    "fallback": "gemini-1.5-flash",
}

# --- Together AI models (OpenAI-compatible endpoint) ---
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
TOGETHER_MODELS = {
    "primary":  "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "fallback":  "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "fallback2": "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "fallback3": "mistralai/Mistral-7B-Instruct-v0.3",
}

# Active model names (resolved from provider)
if LLM_PROVIDER == "nvidia":
    DEFAULT_MODEL = NVIDIA_MODELS["primary"]
    FALLBACK_MODEL = NVIDIA_MODELS["fallback"]
elif LLM_PROVIDER == "gemini":
    DEFAULT_MODEL = GEMINI_MODELS["primary"]
    FALLBACK_MODEL = GEMINI_MODELS["fallback"]
elif LLM_PROVIDER == "together":
    DEFAULT_MODEL = TOGETHER_MODELS["primary"]
    FALLBACK_MODEL = TOGETHER_MODELS["fallback"]
else:  # groq (default)
    DEFAULT_MODEL = GROQ_MODELS["primary"]
    FALLBACK_MODEL = GROQ_MODELS["fallback"]
