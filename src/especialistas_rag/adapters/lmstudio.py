"""Client for the local LM Studio HTTP API."""
import httpx

from .. import settings

_loaded_instances: dict[str, str] = {}
_no_reasoning_models: set[str] = set()
_reasoning_options: dict[str, list[str]] = {}


def headers() -> dict[str, str]:
    token = settings.LMSTUDIO_API_TOKEN
    return {"Authorization": f"Bearer {token}"} if token else {}


def client(timeout=httpx.Timeout(600, connect=5)) -> httpx.Client:
    return httpx.Client(timeout=timeout, trust_env=False, headers=headers())


def models() -> list[dict]:
    try:
        with client(timeout=5) as session:
            response = session.get(settings.LMSTUDIO_URL + "/api/v1/models")
            response.raise_for_status()
            items = response.json()["models"]
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        raise RuntimeError("LM Studio indisponível. Verifique o servidor local e o token.") from exc

    result = []
    for item in items:
        if item.get("type") != "llm" or not item.get("key"):
            continue
        identity = item["key"]
        capabilities = item.get("capabilities") or {}
        reasoning = capabilities.get("reasoning") if isinstance(capabilities, dict) else None
        options = reasoning.get("allowed_options", []) if isinstance(reasoning, dict) else []
        allowed = [option for option in options
                   if option in ("off", "on", "low", "medium", "high")]
        _reasoning_options[identity] = allowed
        if allowed:
            _no_reasoning_models.discard(identity)
        else:
            _no_reasoning_models.add(identity)
        instances = item.get("loaded_instances") or []
        instance_id = next((entry["id"] for entry in instances if entry.get("id")), None)
        if instance_id:
            _loaded_instances[identity] = instance_id
        else:
            _loaded_instances.pop(identity, None)
        result.append({"id": identity, "label": item.get("display_name") or identity,
                       "provider": "lmstudio", "loaded": instance_id is not None,
                       "reasoning_options": allowed})
    return result


def _error_message(response) -> str:
    try:
        return response.json().get("error", {}).get("message", response.text)
    except ValueError:
        return response.text


def complete(model: str, system: str, prompt: str, max_tokens: int, reasoning: str | None = "off") -> str:
    payload = {"model": _loaded_instances.get(model, model), "system_prompt": system, "input": prompt,
               "temperature": 0.1, "max_output_tokens": max_tokens,
               "store": False}
    allowed = _reasoning_options.get(model)
    if reasoning and model not in _no_reasoning_models and (allowed is None or reasoning in allowed):
        payload["reasoning"] = reasoning
    with client() as session:
        response = session.post(settings.LMSTUDIO_URL + "/api/v1/chat", json=payload)
        if response.is_error:
            message = _error_message(response)
            if payload.get("reasoning") == "off" and "does not expose reasoning configuration" in message.lower():
                _no_reasoning_models.add(model)
                payload.pop("reasoning", None)
                response = session.post(settings.LMSTUDIO_URL + "/api/v1/chat", json=payload)
                if response.is_error:
                    message = _error_message(response)
            if response.is_error:
                if "insufficient system resources" in message.lower():
                    raise RuntimeError(
                        f"O LM Studio não conseguiu carregar {model} por falta de recursos. "
                        "Escolha um modelo marcado como carregado ou descarregue outro modelo no LM Studio."
                    )
                raise RuntimeError(f"LM Studio recusou a consulta ({response.status_code}): {message[:400]}")
        result = "\n".join(x.get("content", "") for x in response.json().get("output", []) if x.get("type") == "message").strip()
        if not result:
            raise RuntimeError("O modelo não retornou texto")
        return result
