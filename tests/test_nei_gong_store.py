from fastapi.testclient import TestClient

from especialistas_rag import nei_gong, nei_gong_store


def test_conversation_round_trip_and_continuation(tmp_path):
    path = tmp_path / "chat.sqlite3"
    source = {"citation": "S1", "title": "Book", "author": "Author", "pdf_page": 12,
              "text": "Passage", "id": "book-p12"}
    identity = nei_gong_store.save_turn(path, None, "Primeira pergunta", "Primeira resposta [S1]",
                                        "model-a", [source], {"search_rounds": 1})
    nei_gong_store.save_turn(path, identity, "E depois?", "Segunda resposta",
                             "model-b", [], {"search_rounds": 2})
    listed = nei_gong_store.list_conversations(path)
    assert len(listed) == 1
    assert listed[0]["message_count"] == 4
    assert listed[0]["title"] == "Primeira pergunta"
    saved = nei_gong_store.get_conversation(path, identity)
    assert [m["role"] for m in saved["messages"]] == ["user", "assistant", "user", "assistant"]
    assert saved["messages"][1]["sources"] == [source]
    assert saved["messages"][3]["retrieval"]["search_rounds"] == 2
    assert saved["model"] == "model-b"


def test_chat_api_uses_saved_history_and_returns_same_conversation(tmp_path, monkeypatch):
    monkeypatch.setattr(nei_gong, "CHAT_DB", tmp_path / "chat.sqlite3")
    monkeypatch.setattr(nei_gong, "models", lambda: [{"id": "model-a"}])
    histories = []

    def retrieve(question, model, history):
        histories.append(history)
        return [], {"search_rounds": 1}

    monkeypatch.setattr(nei_gong, "retrieve", retrieve)
    monkeypatch.setattr(nei_gong, "complete", lambda question, model, sources, history, specialist_id, thinking: "Resposta: " + question)
    client = TestClient(nei_gong.app)
    first = client.post("/api/chat", json={"question": "Olá", "model": "model-a"})
    assert first.status_code == 200
    identity = first.json()["conversation_id"]
    second = client.post("/api/chat", json={"question": "Continue", "model": "model-a",
                                        "conversation_id": identity})
    assert second.status_code == 200
    assert second.json()["conversation_id"] == identity
    assert histories[1] == [{"role": "user", "content": "Olá"},
                            {"role": "assistant", "content": "Resposta: Olá"}]
    loaded = client.get(f"/api/conversations/{identity}").json()
    assert len(loaded["messages"]) == 4
    assert client.get("/api/conversations").json()["conversations"][0]["id"] == identity
