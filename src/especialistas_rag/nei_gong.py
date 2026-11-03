"""Compatibility entry point for the original web server and PDF bootstrap."""
import hashlib
import json
import re
import unicodedata
import uuid

from . import core
from .adapters.library import LEGACY_COLLECTION
from .settings import SOURCE_BOOKS
from .web.api import app as app

BOOKS = SOURCE_BOOKS / "nei gong"
DATA = core.DATA / "nei_gong"


def slug(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")[:65]


def catalog() -> list[dict]:
    from pypdf import PdfReader
    result = []
    for path in sorted(BOOKS.glob("*.pdf")):
        reader = PdfReader(path)
        meta = reader.metadata or {}
        title = str(meta.get("/Title") or path.stem).strip()
        author = str(meta.get("/Author") or "").strip()
        if not author and ("Damo Mitchell" in path.name or "Damo-Mitchell" in path.name):
            author = "Damo Mitchell"
        result.append({"id": slug(path.stem), "title": title, "author": author or "Autor não identificado",
                       "filename": path.name, "pages": len(reader.pages), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return result


def ingest() -> dict:
    from pypdf import PdfReader
    from qdrant_client import QdrantClient, models
    books = catalog()
    if not books:
        raise RuntimeError(f"Nenhum PDF em {BOOKS}")
    encoder = core.load_encoder()
    rows = []
    for book in books:
        reader = PdfReader(BOOKS / book["filename"])
        for index, page in enumerate(reader.pages, 1):
            raw = page.extract_text() or ""
            text = re.sub(r"[ \t]+", " ", raw).strip()
            if len(text) < 80:
                continue
            tokens = encoder.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            for number, start in enumerate(range(0, len(tokens), 250)):
                stop = min(start + 320, len(tokens))
                passage = text[tokens[start][0]:tokens[stop-1][1]].strip()
                if len(passage) < 80:
                    continue
                rows.append({"id": f"{book['id']}-p{index:04}-c{number:02}", "book_id": book["id"],
                             "title": book["title"], "author": book["author"], "pdf_page": index,
                             "text": passage})
                if stop == len(tokens):
                    break
    if not rows:
        raise RuntimeError("Nenhum trecho extraído dos PDFs")
    DATA.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(DATA / "qdrant"))
    try:
        if client.collection_exists(LEGACY_COLLECTION):
            client.delete_collection(LEGACY_COLLECTION)
        client.create_collection(LEGACY_COLLECTION, vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
        for start in range(0, len(rows), 64):
            batch = rows[start:start+64]
            vectors = encoder.encode(["passage: " + row["text"] for row in batch], normalize_embeddings=True, batch_size=32)
            client.upsert(LEGACY_COLLECTION, points=[models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, row["id"])), vector=vector.tolist(), payload=row)
                for row, vector in zip(batch, vectors)])
            if start % 1280 == 0:
                print(f"Nei Gong: {min(start+64, len(rows))}/{len(rows)} trechos", flush=True)
    finally:
        client.close()
    core.write_jsonl(DATA / "chunks.jsonl", rows)
    (DATA / "books.json").write_text(json.dumps(books, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"books": len(books), "chunks": len(rows), "collection": LEGACY_COLLECTION}
    (DATA / "index.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
