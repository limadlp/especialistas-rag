import pytest
from especialistas_rag.core import fuse, lexical, chunks_from_pages


def test_fusion_prefers_agreement_and_does_not_duplicate():
    assert [x[0] for x in fuse(["a", "b"], ["b", "c"])] == ["b", "a", "c"]


def test_portuguese_accent_normalization():
    assert lexical("Punição e EXTINÇÃO") == ["punicao", "extincao"]


def test_chunking_preserves_original_text_and_source():
    class Tokenizer:
        def __call__(self, text, **kwargs):
            import re
            return {"offset_mapping": [(m.start(),m.end()) for m in re.finditer(r"\S+",text)]}
    text = " ".join(f"palavra{i}" for i in range(100))
    p = {"id":"pdf020-left", "pdf_page":20, "side":"left", "text":text}
    chunks = chunks_from_pages([p], Tokenizer(), size=30, overlap=5)
    assert chunks[0]["text"] in text and chunks[-1]["text"].endswith("palavra99")
    assert all(c["pdf_page"] == 20 and c["token_count"] <= 30 for c in chunks)
    assert chunks[0]["text"].split()[-5:] == chunks[1]["text"].split()[:5]
    with pytest.raises(ValueError): chunks_from_pages([p], Tokenizer(), size=10, overlap=10)


def test_api_rejects_unknown_model_and_excessive_history():
    from fastapi.testclient import TestClient
    from especialistas_rag.api import app
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/answer", json={"question":"teste", "model":"desconhecido"}).status_code == 422
        assert client.post("/search", json={"question":"teste", "limit":100}).status_code == 422


def test_reasoning_is_rejected_even_if_message_present(monkeypatch):
    from especialistas_rag import core
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"output":[{"type":"message","content":"x"}],"stats":{"reasoning_output_tokens":3}}
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, *args, **kwargs):
            assert kwargs['json']['reasoning'] == 'off'
            assert kwargs['json']['store'] is False
            return Response()
    monkeypatch.setattr(core.httpx,"Client",Client)
    with pytest.raises(RuntimeError,match="Thinking"): core.answer("x","pequeno")
