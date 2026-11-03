"""Model discovery and the local chat default."""
from .lmstudio import complete, models


def default_model(available: list[dict]) -> dict | None:
    return next((entry for entry in available
                 if entry.get("provider") == "lmstudio" and entry.get("loaded")), None)
