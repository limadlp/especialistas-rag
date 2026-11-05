"""Model discovery, local defaults, and provider dispatch."""
from . import gemini, lmstudio


def models() -> list[dict]:
    hosted = gemini.models()
    try:
        local = lmstudio.models()
    except RuntimeError:
        if not hosted:
            raise
        local = []
    return local + hosted


def default_model(available: list[dict]) -> dict | None:
    return next((entry for entry in available
                 if entry.get("provider") == "lmstudio" and entry.get("loaded")), None)


def complete(model: str, system: str, prompt: str, max_tokens: int,
             reasoning: str | None = "off") -> str:
    if model.startswith(gemini.PREFIX):
        return gemini.complete(model.removeprefix(gemini.PREFIX), system, prompt, max_tokens)
    return lmstudio.complete(model, system, prompt, max_tokens, reasoning)
