"""Measure retrieval on a disposable library, with an optional paired local LLM run."""
import argparse
import json
import os
from pathlib import Path
import re
from statistics import mean
from tempfile import TemporaryDirectory
from xml.sax.saxutils import escape
from zipfile import ZipFile

from .cases import CASES, DOCUMENTS

CITATION = re.compile(r"\[(S\d+)\]")
SYSTEM = (
    "Responda em português, de forma direta. Se houver fontes, fundamente a resposta nelas "
    "e use os códigos de citação fornecidos, como [S1]. Se a informação for insuficiente, "
    "indique o que falta. Não invente fatos nem referências."
)


def write_epub(path: Path, title: str, text: str) -> None:
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
              <manifest><item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/></manifest>
              <spine><itemref idref="chapter"/></spine>
            </package>""")
        archive.writestr("chapter.xhtml", f'<html xmlns="http://www.w3.org/1999/xhtml"><body><p>{escape(text)}</p></body></html>')


def score_answer(answer: str, expected: tuple[str, ...]) -> bool:
    return any(re.search(r"(?<!\w)" + re.escape(variant.casefold()) + r"(?!\w)", answer.casefold())
               for variant in expected)


def evaluate(embedding_model: Path, generate: bool) -> dict:
    with TemporaryDirectory(prefix="especialistas-rag-eval-") as target:
        # Configure paths before loading the application so the evaluation cannot open user data.
        os.environ["TAC_DATA"] = target
        os.environ["TAC_EMBEDDING_MODEL"] = str(embedding_model.resolve())
        from especialistas_rag import rag
        from especialistas_rag.adapters import documents, llm, lmstudio, specialists
        from especialistas_rag.adapters.library import library

        specialist = specialists.create("Projeto Cedro", "Operações do arquivo e do viveiro")
        identity = specialist["id"]
        specialists.set_system_prompt(identity, SYSTEM)
        model = llm.default_model(lmstudio.models()) if generate else None
        if generate and model is None:
            raise RuntimeError("Carregue um modelo no LM Studio antes de usar --generate.")
        index = library(identity)
        rows = []
        try:
            for key, document in DOCUMENTS.items():
                path = Path(target) / f"{key}.epub"
                write_epub(path, document["title"], document["text"])
                metadata = documents.book_metadata(path, path.name, "por")
                index.add_book(path, {**metadata, "id": key})

            for question, expected_book, aliases in CASES:
                candidates = index.search([question], limit=3)
                ranks = [rank for rank, item in enumerate(candidates, 1) if item["book_id"] == expected_book]
                result = {"question": question, "expected_book": expected_book,
                          "rank_at_3": ranks[0] if ranks else None}
                if model:
                    # Same question, system prompt, and model in both conditions.
                    baseline = rag.complete(question, model, [], [], identity)
                    selected, _ = rag.retrieve(question, model["id"], [], identity)
                    sources = [{**item, "citation": f"S{number}"} for number, item in enumerate(selected, 1)]
                    answer = rag.complete(question, model, sources, [], identity)
                    cited = set(CITATION.findall(answer))
                    valid = {item["citation"] for item in sources}
                    gold = {item["citation"] for item in sources if item["book_id"] == expected_book}
                    result.update(
                        baseline_match=score_answer(baseline, aliases),
                        rag_match=score_answer(answer, aliases),
                        rag_valid_citations=bool(cited) and cited <= valid,
                        rag_cites_expected_book=bool(cited & gold),
                    )
                rows.append(result)
        finally:
            index.close()

    ranks = [row["rank_at_3"] for row in rows]
    report = {
        "questions": len(rows),
        "hit_at_1": mean(rank == 1 for rank in ranks),
        "hit_at_3": mean(rank is not None for rank in ranks),
        "mrr_at_3": mean(1 / rank if rank else 0 for rank in ranks),
        "cases": rows,
    }
    if model:
        report["model"] = model["id"]
        for key in ("baseline_match", "rag_match", "rag_valid_citations", "rag_cites_expected_book"):
            report[key] = mean(row[key] for row in rows)
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
