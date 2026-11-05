import httpx
from fastapi.testclient import TestClient

from especialistas_rag import rag, settings
from especialistas_rag.adapters import llm, lmstudio, specialists
from especialistas_rag.web.api import app


def test_chat_defaults_to_loaded_lmstudio_and_selects_gemini_explicitly(tmp_path, monkeypatch):
    monkeypatch.setattr(specialists, "REGISTRY", tmp_path / "registry.json")
    monkeypatch.setattr(specialists, "DATA_ROOT", tmp_path / "specialists")
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setattr(settings, "GOOGLE_MODEL", "test-gemini")
    monkeypatch.setattr(lmstudio, "models", lambda: [
        {"id": "downloaded", "provider": "lmstudio", "loaded": False},
        {"id": "running", "provider": "lmstudio", "loaded": True},
    ])
    monkeypatch.setattr(rag, "retrieve", lambda *args: ([], {"search_rounds": 1}))
    monkeypatch.setattr(lmstudio, "complete", lambda *args: "Resposta local")
    calls = []

    def send(url, *, headers, json, timeout):
        calls.append(json)
        assert url == llm.gemini.ENDPOINT
        assert headers == {"Authorization": "Bearer test-google-key"}
        return httpx.Response(200, request=httpx.Request("POST", url),
                              json={"choices": [{"message": {"content": "Resposta Gemini"}}]})

    monkeypatch.setattr(httpx, "post", send)
    with TestClient(app) as client:
        models = client.get("/api/models").json()
        assert models["default_model"] == "running"
        assert models["models"][-1]["id"] == "gemini:test-gemini"

        payload = {"question": "Uma pergunta fictícia", "specialist_id": "psychology"}
        local = client.post("/api/chat", json=payload)
        assert local.status_code == 200
        assert local.json()["answer"] == "Resposta local"
        assert calls == []

        def offline():
            raise RuntimeError("LM Studio indisponível")

        monkeypatch.setattr(lmstudio, "models", offline)
        assert client.get("/api/models").json()["default_model"] is None
        assert client.post("/api/chat", json=payload).status_code == 422
        assert calls == []
        hosted = client.post("/api/chat", json={**payload, "model": "gemini:test-gemini"})
        assert hosted.status_code == 200
        assert hosted.json()["answer"] == "Resposta Gemini"
        assert calls[0]["model"] == "test-gemini"
        assert payload["question"] in calls[0]["messages"][1]["content"]


def test_lmstudio_discovery_authenticates_and_excludes_embedding_models(monkeypatch):
    monkeypatch.setattr(settings, "LMSTUDIO_API_TOKEN", "test-lm-token")
    def reply(request):
        assert request.url.path == "/api/v1/models"
        assert request.headers["Authorization"] == "Bearer test-lm-token"
        return httpx.Response(200, json={"models": [
            {"key": "test-llm", "type": "llm", "loaded_instances": [{"id": "loaded-id"}]},
            {"key": "test-embedder", "type": "embedding"},
        ]})

    monkeypatch.setattr(lmstudio, "client", lambda timeout: httpx.Client(
        transport=httpx.MockTransport(reply), headers=lmstudio.headers(), timeout=timeout,
    ))
    models = llm.models()
    assert llm.default_model(models)["id"] == "test-llm"
    assert "test-embedder" not in {model["id"] for model in models}
