"""Google Gemini adapter through its OpenAI-compatible chat endpoint."""
import httpx

from .. import settings

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
PREFIX = "gemini:"


def models() -> list[dict]:
    if not settings.GOOGLE_API_KEY or not settings.GOOGLE_MODEL:
        return []
    return [{"id": PREFIX + settings.GOOGLE_MODEL, "label": settings.GOOGLE_MODEL,
             "provider": "gemini", "loaded": False, "reasoning_options": []}]


def complete(model: str, system: str, prompt: str, max_tokens: int) -> str:
    if not settings.GOOGLE_API_KEY or model != settings.GOOGLE_MODEL:
        raise RuntimeError("Configure GOOGLE_API_KEY e TAC_GOOGLE_MODEL para usar o Gemini.")
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    try:
        response = httpx.post(
            ENDPOINT, headers={"Authorization": f"Bearer {settings.GOOGLE_API_KEY}"},
            json=body, timeout=httpx.Timeout(120, connect=5),
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Gemini recusou a consulta (HTTP {exc.response.status_code}).") from exc
    except httpx.RequestError as exc:
        raise RuntimeError("Não foi possível conectar à API do Gemini.") from exc
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Gemini não retornou uma resposta de texto.") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Gemini não retornou uma resposta de texto.")
    return content.strip()
