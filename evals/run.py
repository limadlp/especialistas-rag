"""Measure retrieval on a disposable library, with an optional paired local LLM run."""
import argparse
from collections import Counter
from hashlib import file_digest
import json
import os
from pathlib import Path
import re
from statistics import mean
from tempfile import TemporaryDirectory
from xml.sax.saxutils import escape
from zipfile import ZipFile

from .cases import CASES, DOCUMENTS

CUTOFFS = (1, 3, 5)
ABSTENTION = "SEM EVIDÊNCIA"
CITATION = re.compile(r"\[(S\d+)\]")
SYSTEM = (
    "Responda em português, de forma direta. Se houver fontes, fundamente a resposta nelas "
    "e use os códigos de citação fornecidos, como [S1]. Se a informação for insuficiente, "
    f"responda apenas {ABSTENTION}. Não invente fatos nem referências."
)


def write_epub(path: Path, title: str, sections: tuple[tuple[str, str], ...]) -> None:
    manifest = "".join(f'<item id="s{i}" href="s{i}.xhtml" media-type="application/xhtml+xml"/>'
                       for i in range(1, len(sections) + 1))
    spine = "".join(f'<itemref idref="s{i}"/>' for i in range(1, len(sections) + 1))
    with ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", """
            <container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
              <rootfiles><rootfile full-path="book.opf" media-type="application/oebps-package+xml"/></rootfiles>
            </container>""")
        archive.writestr("book.opf", f"""
            <package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
              <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                <dc:title>{escape(title)}</dc:title><dc:creator>Projeto Cedro</dc:creator>
                <dc:identifier id="id">cedro:{escape(path.stem)}</dc:identifier>
                <dc:language>pt</dc:language>
              </metadata>
              <manifest>{manifest}</manifest>
              <spine>{spine}</spine>
            </package>""")
        for number, (heading, text) in enumerate(sections, 1):
            archive.writestr(f"s{number}.xhtml",
                             f'<html xmlns="http://www.w3.org/1999/xhtml"><body>'
                             f'<h1>{escape(heading)}</h1><p>{escape(text)}</p></body></html>')


def address(passage: dict) -> str:
    return f"{passage['book_id']}:{passage['section_index']}"


def score_answer(answer: str, groups: tuple[str, ...]) -> bool:
    return all(any(re.search(r"(?<!\w)" + re.escape(alias.casefold()) + r"(?!\w)", answer.casefold())
                   for alias in group.split("|")) for group in groups)


def retrieval_metrics(rows: list[dict]) -> dict:
    """Score answerable cases only. Multiple evidence passages count separately for recall."""
    rows = [row for row in rows if row["expected_passages"]]
    metrics = {"questions": len(rows)}
    expected = [set(row["expected_passages"]) for row in rows]
    for k in CUTOFFS:
        found = [gold.intersection(row["retrieved_passages"][:k]) for gold, row in zip(expected, rows)]
        metrics[f"hit_at_{k}"] = sum(bool(passages) for passages in found)
        metrics[f"all_evidence_at_{k}"] = sum(actual == gold for actual, gold in zip(found, expected))
        metrics[f"recall_at_{k}"] = mean(len(actual) / len(gold) for actual, gold in zip(found, expected))
    metrics["mrr_at_5"] = mean(
        next((1 / rank for rank, address in enumerate(row["retrieved_passages"][:5], 1)
              if address in row["expected_passages"]), 0)
        for row in rows
    )
    return metrics


def generation_metrics(case, rag, model: dict, identity: str) -> dict:
    baseline = rag.complete(case.question, model, [], [], identity)
    selected, _ = rag.retrieve(case.question, model["id"], [], identity)
    sources = [{**item, "citation": f"S{number}"} for number, item in enumerate(selected, 1)]
    answer = rag.complete(case.question, model, sources, [], identity)
    if not case.evidence:
        return {"baseline_abstained": baseline.strip().upper() == ABSTENTION,
                "rag_abstained": answer.strip().upper() == ABSTENTION}
    cited = set(CITATION.findall(answer))
    valid = {item["citation"] for item in sources}
    cited_passages = {address(item) for item in sources if item["citation"] in cited}
    return {
        "baseline_match": score_answer(baseline, case.answers),
        "rag_match": score_answer(answer, case.answers),
        "rag_valid_citations": bool(cited) and cited <= valid,
        "rag_cites_all_evidence": set(case.evidence) <= cited_passages,
    }


def evaluate(embedding_model: Path, generate: bool) -> dict:
    with TemporaryDirectory(prefix="especialistas-rag-eval-") as target:
        # Configure paths before loading the application so the evaluation cannot open user data.
        os.environ["TAC_DATA"] = target
        os.environ["TAC_EMBEDDING_MODEL"] = str(embedding_model.resolve())
        from especialistas_rag import rag
        from especialistas_rag.adapters import documents, llm, lmstudio, specialists
        from especialistas_rag.adapters.library import library

        specialist = specialists.create("Projeto Cedro", "Arquivo, transporte, viveiro e auditoria")
        identity = specialist["id"]
        specialists.set_system_prompt(identity, SYSTEM)
        model = llm.default_model(lmstudio.models()) if generate else None
        if generate and model is None:
            raise RuntimeError("Carregue um modelo no LM Studio antes de usar --generate.")
        index = library(identity)
        rows = []
        try:
            for key, (title, sections) in DOCUMENTS.items():
                path = Path(target) / f"{key}.epub"
                write_epub(path, title, sections)
                metadata = documents.book_metadata(path, path.name, "por")
                index.add_book(path, {**metadata, "id": key})
            observed = Counter(address(item) for item in index.rows)
            expected = Counter(f"{key}:{number}" for key, (_, sections) in DOCUMENTS.items()
                               for number in range(1, len(sections) + 1))
            if observed != expected:
                raise ValueError("The benchmark requires exactly one indexed passage per EPUB section.")
            for case in CASES:
                if not set(case.evidence) <= observed.keys():
                    raise ValueError(f"Unknown evidence address in {case.id}.")
                candidates = index.search([case.question], limit=max(CUTOFFS))
                rows.append({
                    "id": case.id,
                    "kind": case.kind,
                    "question": case.question,
                    "expected_passages": case.evidence,
                    "retrieved_passages": [address(item) for item in candidates],
                })
                if model:
                    rows[-1]["generation"] = generation_metrics(case, rag, model, identity)
        finally:
            index.close()

    with (embedding_model / "model.safetensors").open("rb") as weights:
        sha256 = file_digest(weights, "sha256").hexdigest()
    report = {
        "benchmark": "cedro-v2",
        "documents": len(DOCUMENTS),
        "passages": len(observed),
        "questions": len(rows),
        "embedding_weights_sha256": sha256,
        "retrieval": retrieval_metrics(rows),
        "by_kind": {kind: retrieval_metrics([row for row in rows if row["kind"] == kind])
                    for kind in sorted({case.kind for case in CASES if case.evidence})},
        "cases": rows,
    }
    if model:
        counts = Counter()
        for row in rows:
            for metric, value in row["generation"].items():
                counts[metric] += value
        report["generation"] = {"model": model["id"], "counts": dict(counts),
                                "answerable": report["retrieval"]["questions"],
                                "unanswerable": len(rows) - report["retrieval"]["questions"]}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-model", type=Path,
                        default=Path("data/models/multilingual-e5-small"))
    parser.add_argument("--generate", action="store_true", help="Compare answers with and without RAG using LM Studio")
    args = parser.parse_args()
    if not args.embedding_model.is_dir():
        parser.error("Pass the local multilingual-e5-small directory through --embedding-model.")
    print(json.dumps(evaluate(args.embedding_model, args.generate), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
