from fastapi.testclient import TestClient

from especialistas_rag import nei_gong, specialist_platform


def test_specialists_have_isolated_conversations(tmp_path, monkeypatch):
    monkeypatch.setattr(specialist_platform, "REGISTRY", tmp_path / "registry.json")
    monkeypatch.setattr(specialist_platform, "DATA_ROOT", tmp_path / "specialists")
    monkeypatch.setattr(nei_gong, "models", lambda: [{"id": "local-model"}])
    monkeypatch.setattr(nei_gong, "retrieve", lambda question, model, history, identity: ([], {"search_rounds": 1}))
    monkeypatch.setattr(nei_gong, "complete", lambda question, model, sources, history, identity, thinking: "Resposta local")
    client = TestClient(nei_gong.app)

    alpha = client.post("/api/specialists", json={"name": "Alpha", "description": "Primeiro"}).json()
    beta = client.post("/api/specialists", json={"name": "Beta", "description": "Segundo"}).json()
    first = client.post("/api/chat", json={"specialist_id": alpha["id"], "model": "local-model", "question": "Teste"})
    assert first.status_code == 200
    conversation_id = first.json()["conversation_id"]
    assert len(client.get(f"/api/specialists/{alpha['id']}/conversations").json()["conversations"]) == 1
    assert client.get(f"/api/specialists/{beta['id']}/conversations").json()["conversations"] == []
    assert client.post("/api/chat", json={"specialist_id": beta["id"], "model": "local-model", "question": "Continuação", "conversation_id": conversation_id}).status_code == 404


def test_registry_rejects_unsupported_ocr_language(tmp_path, monkeypatch):
    monkeypatch.setattr(specialist_platform, "REGISTRY", tmp_path / "registry.json")
    client = TestClient(nei_gong.app)
    response = client.post("/api/specialists", json={"name": "Teste", "ocr_language": "xyz"})
    assert response.status_code == 422
    assert [entry["id"] for entry in client.get("/api/specialists").json()["specialists"]] == ["nei-gong", "psychology"]


def test_system_prompt_can_be_saved_and_restored(tmp_path, monkeypatch):
    monkeypatch.setattr(specialist_platform, "REGISTRY", tmp_path / "registry.json")
    client = TestClient(nei_gong.app)
    original = client.get("/api/specialists/nei-gong/prompt").json()
    assert original["custom"] is False
    saved = client.put("/api/specialists/nei-gong/prompt", json={"system_prompt": "Use os livros."}).json()
    assert saved == {"system_prompt": "Use os livros.", "custom": True}
    restored = client.put("/api/specialists/nei-gong/prompt", json={"system_prompt": None}).json()
    assert restored == original


def test_password_protected_pdf_upload(tmp_path, monkeypatch):
    from io import BytesIO
    from urllib.parse import quote
    from pypdf import PdfWriter

    monkeypatch.setattr(specialist_platform, "REGISTRY", tmp_path / "registry.json")
    monkeypatch.setattr(specialist_platform, "DATA_ROOT", tmp_path / "specialists")
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("senha ç", algorithm="AES-256")
    data = BytesIO()
    writer.write(data)
    client = TestClient(nei_gong.app)
    headers = {"content-type": "application/pdf", "x-filename": "livro.pdf"}
    response = client.post("/api/specialists/nei-gong/books", content=data.getvalue(), headers=headers)
    assert response.status_code == 422
    assert "Senha" in response.json()["detail"]
    headers["x-pdf-password"] = quote("senha ç")
    captured = {}
    class Library:
        def add_book(self, path, book, progress, password):
            captured.update(password=password, book=book)
            return book
    monkeypatch.setattr(nei_gong, "library", lambda identity: Library())
    response = client.post("/api/specialists/nei-gong/books", content=data.getvalue(), headers=headers)
    assert response.status_code == 202
    import time
    for _ in range(100):
        job = client.get(f"/api/jobs/{response.json()['job_id']}").json()
        if job["status"] == "done":
            break
        time.sleep(0.01)
    assert job["status"] == "done"
    assert captured["password"] == "senha ç"
    assert "senha" not in str(response.json()).lower()
    assert "password" not in captured["book"]


def test_protected_pdf_ocr_uses_password(tmp_path, monkeypatch):
    from pypdf import PdfWriter
    import pypdfium2

    path = tmp_path / "protected.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("secret", algorithm="AES-256")
    writer.write(path)
    real_document = pypdfium2.PdfDocument
    seen = []
    def document(*args, **kwargs):
        seen.append(kwargs.get("password"))
        return real_document(*args, **kwargs)
    monkeypatch.setattr(pypdfium2, "PdfDocument", document)
    monkeypatch.setattr(specialist_platform, "ocr_image", lambda image, language: "texto")
    pages, count = specialist_platform.extract_pages(path, "por", password="secret")
    assert count == 1 and pages[0]["text"] == "texto"
    assert seen == ["secret"]
