import os

CLOSEAI_CHAT_URL = "https://api.openai-proxy.org/v1/chat/completions"
CLOSEAI_MODELS_URL = "https://api.openai-proxy.org/api/v1/management/models"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 2048
DEFAULT_FULL_RUN_HARD_STOP_RMB = 10000.0


def _get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_optional_env(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


def get_closeai_chat_url() -> str:
    return _get_optional_env("CLOSEAI_CHAT_URL", CLOSEAI_CHAT_URL)


def get_closeai_models_url() -> str:
    return _get_optional_env("CLOSEAI_MODELS_URL", CLOSEAI_MODELS_URL)


def get_closeai_api_key() -> str:
    return _get_required_env("CLOSEAI_API_KEY")


def get_closeai_admin_key() -> str:
    return _get_required_env("CLOSEAI_ADMIN_KEY")
