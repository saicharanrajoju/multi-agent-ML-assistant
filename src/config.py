import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
E2B_API_KEY = os.getenv("E2B_API_KEY")

required_keys = [
    "GROQ_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "E2B_API_KEY",
]

missing_keys = [key for key in required_keys if not os.getenv(key)]

if missing_keys:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")

# Set LANGSMITH_TRACING to true explicitly
os.environ["LANGSMITH_TRACING"] = "true"

# LLM Model Configuration
LLM_MODELS = [
    {"name": "llama-3.3-70b-versatile", "max_tokens": 4096, "description": "Primary - best quality"},
    {"name": "llama-3.1-8b-instant", "max_tokens": 4096, "description": "Fallback - fast and cheap"},
]

DEFAULT_MODEL = LLM_MODELS[0]["name"]
FALLBACK_MODEL = LLM_MODELS[1]["name"]
