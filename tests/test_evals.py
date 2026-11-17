from collections import Counter

from pytest import approx

from evals.cases import CASES, DOCUMENTS
from evals.run import generation_metrics, retrieval_metrics, score_answer


def test_corpus_labels_point_to_real_sections():
    assert len({case.id for case in CASES}) == len(CASES)
    assert Counter(case.kind for case in CASES) == dict.fromkeys(
        ["direct", "paraphrase", "confusable", "multi_evidence", "unanswerable"], 6
    )
    for case in CASES:
        for address in case.evidence:
            book, section = address.split(":")
            assert 1 <= int(section) <= len(DOCUMENTS[book][1])
        assert bool(case.answers) == bool(case.evidence)


def test_hit_all_evidence_and_reciprocal_rank_count_separately():
    rows = [
        {"expected_passages": ["a:1", "b:2"], "retrieved_passages": ["b:1", "a:1", "a:2", "b:2"]},
        {"expected_passages": ["a:2"], "retrieved_passages": ["b:1"]},
        {"expected_passages": [], "retrieved_passages": ["a:1"]},
    ]
    result = retrieval_metrics(rows)
    assert result["questions"] == 2
    assert result["hit_at_3"] == 1
    assert result["all_evidence_at_3"] == 0
    assert result["recall_at_3"] == approx(0.25)
    assert result["all_evidence_at_5"] == 1
    assert result["mrr_at_5"] == approx(0.25)


def test_numeric_variants_do_not_match_substrings():
    assert score_answer("Doze horas: 8mm no piloto e 12 mm depois.", ("8 mm|8mm", "12 mm|12mm"))
    assert not score_answer("18 mm no piloto e 12 mm depois.", ("8 mm|8mm", "12 mm|12mm"))


def test_generation_requires_all_expected_citations_and_checks_abstention():
    class Rag:
        def complete(self, question, model, sources, history, identity):
            return "42, 48, 30 e 36 [S1]" if sources and question.startswith("Quais faixas") else "SEM EVIDÊNCIA."

        def retrieve(self, *args):
            return [{"book_id": "papel", "section_index": 1}, {"book_id": "fotos", "section_index": 1}], {}

    multi = next(case for case in CASES if case.id == "m01")
    missing = next(case for case in CASES if case.id == "n01")
    model = {"id": "test"}
    result = generation_metrics(multi, Rag(), model, "test")
    assert result["rag_match"] and result["rag_valid_citations"]
    assert not result["rag_cites_all_evidence"]
    assert not result["rag_match_with_all_citations"]
    assert generation_metrics(missing, Rag(), model, "test")["rag_abstained"]

    class GroupedCitations(Rag):
        def complete(self, question, model, sources, history, identity):
            return "42, 48, 30 e 36 [S1, S2]" if sources else "SEM EVIDÊNCIA."

    grouped = generation_metrics(multi, GroupedCitations(), model, "test")
    assert grouped["rag_match_with_all_citations"]
