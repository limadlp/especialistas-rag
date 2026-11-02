import pytest

from especialistas_rag import nei_gong


def test_rerank_parser_keeps_only_known_unique_candidates():
    available = {"C1": {"id": "one"}, "C2": {"id": "two"}}
    selected, query = nei_gong.parse_rerank(
        '```json\n{"ids":["C2","C2","C99","C1"],"followup_query":"meditation attention"}\n```',
        available,
    )
    assert [item["id"] for item in selected] == ["two", "one"]
    assert query == "meditation attention"


def test_rerank_parser_ignores_extra_json_after_first_result():
    available = {"C1": {"id": "one"}}
    selected, query = nei_gong.parse_rerank(
        '{"ids":["C1"],"followup_query":""}\n{"note":"extra"}', available)
    assert selected == [available["C1"]]
    assert query == ""


def test_rerank_falls_back_to_retrieved_passages_on_malformed_json(monkeypatch):
    candidates = [{"id": "one", "author": "A", "title": "Livro", "pdf_page": 1, "text": "Trecho"}]
    monkeypatch.setattr(nei_gong, "lm_chat", lambda *args: '{invalid json}')
    selected, followup = nei_gong.rerank("Pergunta", "modelo", candidates, True)
    assert selected == candidates
    assert followup == ""


def test_retrieve_runs_one_followup_and_reranks_combined_pool(monkeypatch):
    def row(identity):
        return {"id": identity, "book_id": identity, "title": identity, "author": identity, "pdf_page": 30}

    first = [row("a"), row("b")]
    second = [row("b"), row("c")]
    searches = []
    reranks = []

    class LibraryStub:
        class Encoder:
            def encode(self, values, normalize_embeddings):
                return [[1.0, 0.0], [0.0, 1.0]]

        encoder = Encoder()

        def search(self, queries, limit):
            searches.append((queries, limit))
            return first if len(searches) == 1 else second

    def rerank_stub(question, model, candidates, allow_followup):
        reranks.append((question, [x["id"] for x in candidates], allow_followup))
        return ([first[0]], "second angle") if allow_followup else ([second[1]], "")

    monkeypatch.setattr(nei_gong, "library", lambda: LibraryStub())
    monkeypatch.setattr(nei_gong, "rerank", rerank_stub)
    selected, info = nei_gong.retrieve("Compare attention and breath", "model", [])
    assert {x["id"] for x in selected} == {"a", "b", "c"}
    assert searches == [(["Compare attention and breath"], 24), (["second angle"], 16)]
    assert reranks[1][1] == ["a", "b", "c"]
    assert info["search_rounds"] == 2


def test_comparison_retains_buddhist_books():
    def row(book):
        return {"id": book, "book_id": book, "title": book, "author": "Author", "pdf_page": 50}

    daoist = row("daoist-nei-gong")
    mind = row("the-mind-illuminated")
    jhanas = row("right-concentration")
    selected = nei_gong.ensure_comparison_sources(
        "Compare a meditação budista com o Nei Gong", [daoist], [daoist, mind, jhanas], limit=8)
    assert {x["book_id"] for x in selected} == {"daoist-nei-gong", "the-mind-illuminated", "right-concentration"}


def test_comparison_keeps_both_buddhist_books_when_reranker_fills_limit():
    daoist = [{"id": f"d{i}", "book_id": "daoist-nei-gong", "title": "Daoist Nei Gong",
               "author": "Damo Mitchell", "pdf_page": i + 20} for i in range(8)]
    mind = {"id": "m", "book_id": "the-mind-illuminated", "title": "The Mind Illuminated",
            "author": "Culadasa", "pdf_page": 100}
    jhanas = {"id": "j", "book_id": "right-concentration", "title": "Right Concentration",
              "author": "Leigh Brasington", "pdf_page": 80}
    result = nei_gong.ensure_comparison_sources(
        "Compare meditação budista e Nei Gong", daoist, daoist + [mind, jhanas])
    assert len(result) == 8
    assert {"the-mind-illuminated", "right-concentration"}.issubset({x["book_id"] for x in result})


def test_lm_chat_reuses_loaded_instance_configuration(monkeypatch):
    sent = {}

    class Response:
        is_error = False

        def json(self):
            return {"output": [{"type": "message", "content": "Resposta"}]}

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json):
            sent.update(json)
            return Response()

    monkeypatch.setattr(nei_gong.httpx, "Client", Client)
    assert nei_gong.lm_chat("model", "system", "question", 100) == "Resposta"
    assert "context_length" not in sent


def test_model_reasoning_options_select_supported_value():
    assert nei_gong.reasoning_setting({"reasoning_options": ["off", "on"]}, True) == "on"
    assert nei_gong.reasoning_setting({"reasoning_options": ["off", "low", "medium"]}, True) == "medium"
    assert nei_gong.reasoning_setting({"reasoning_options": ["off", "on"]}, False) == "off"
    assert nei_gong.reasoning_setting({"reasoning_options": ["on"]}, False) is None


def test_thinking_uses_supported_on_setting(monkeypatch):
    payloads = []

    class Response:
        is_error = False

        def json(self):
            return {"output": [{"type": "message", "content": "Resposta"}]}

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json):
            payloads.append(json)
            return Response()

    monkeypatch.setattr(nei_gong.httpx, "Client", Client)
    monkeypatch.setitem(nei_gong._reasoning_options, "thinking-model", ["on"])
    nei_gong._no_reasoning_models.discard("thinking-model")
    assert nei_gong.lm_chat("thinking-model", "Sistema", "Pergunta", 100, "on") == "Resposta"
    assert payloads[0]["reasoning"] == "on"
    nei_gong.lm_chat("thinking-model", "Sistema", "Pergunta", 100, "off")
    assert "reasoning" not in payloads[1]


def test_lm_chat_explains_resource_guardrail(monkeypatch):
    class Response:
        is_error = True
        status_code = 400
        text = ""

        def json(self):
            return {"error": {"message": "Model loading was stopped due to insufficient system resources"}}

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json):
            return Response()

    monkeypatch.setattr(nei_gong.httpx, "Client", Client)
    with pytest.raises(RuntimeError, match="falta de recursos"):
        nei_gong.lm_chat("model", "system", "question", 100)


def test_lm_chat_retries_without_reasoning_when_model_rejects_it(monkeypatch):
    monkeypatch.setattr(nei_gong, "_no_reasoning_models", set())
    payloads = []

    class ErrorResponse:
        is_error = True
        status_code = 400
        text = ""

        def json(self):
            return {"error": {"message": "Model 'test-model' does not expose reasoning configuration."}}

    class OkResponse:
        is_error = False

        def json(self):
            return {"output": [{"type": "message", "content": "Resposta sem reasoning"}]}

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json):
            payloads.append(dict(json))
            return ErrorResponse() if len(payloads) == 1 else OkResponse()

    monkeypatch.setattr(nei_gong.httpx, "Client", Client)
    model_id = "test-model"
    assert nei_gong.lm_chat(model_id, "system", "question", 100) == "Resposta sem reasoning"
    assert payloads[0]["reasoning"] == "off"
    assert "reasoning" not in payloads[1]
    assert model_id in nei_gong._no_reasoning_models
