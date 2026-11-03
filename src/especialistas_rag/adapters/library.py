"""Per-specialist dense and lexical index, backed by Qdrant and BM25."""
from pathlib import Path
import re
import threading
import uuid

import numpy as np

from .. import retrieval
from ..files import read_jsonl
from . import documents, specialists

LEGACY_COLLECTION = "nei_gong_books"
_libraries: dict[str, "Library"] = {}
_library_lock = threading.Lock()


def unit_key(row: dict) -> tuple:
    if "section_index" in row:
        return row["book_id"], row["section_index"], row.get("chunk_index", 0) // 3
    return row["book_id"], row["pdf_page"]


class Library:
    def __init__(self, identity: str = "nei-gong"):
        from qdrant_client import QdrantClient, models
        from rank_bm25 import BM25Okapi
        self.identity = identity
        self.folder = specialists.data_dir(identity)
        self.collection = LEGACY_COLLECTION if identity == "nei-gong" else "books"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.folder / "qdrant"))
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(self.collection, vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
        chunks_path = self.folder / "chunks.jsonl"
        self.rows = read_jsonl(chunks_path) if chunks_path.exists() else []
        self.books = specialists.books(identity)
        self.page_limits = {book["id"]: book.get("pages", 0) for book in self.books}
        self.by_id = {row["id"]: row for row in self.rows}
        self.bm25 = BM25Okapi([retrieval.lexical(row["text"]) for row in self.rows]) if self.rows else None
        self.lock = threading.Lock()

    @property
    def encoder(self):
        return retrieval.load_encoder()

    def refresh_lexical(self):
        from rank_bm25 import BM25Okapi
        self.by_id = {row["id"]: row for row in self.rows}
        self.page_limits = {book["id"]: book.get("pages", 0) for book in self.books}
        self.bm25 = BM25Okapi([retrieval.lexical(row["text"]) for row in self.rows]) if self.rows else None

    def add_book(self, pdf: Path, book: dict, progress=None, password: str = "") -> dict:
        from qdrant_client import models
        rows = documents.chunks_for_book(pdf, book, self.encoder, progress, password)
        with self.lock:
            if any(item["sha256"] == book["sha256"] for item in self.books):
                raise ValueError("Este livro já está na biblioteca deste especialista")
            inserted = []
            try:
                for start in range(0, len(rows), 64):
                    batch = rows[start:start+64]
                    vectors = self.encoder.encode(["passage: " + row["text"] for row in batch], normalize_embeddings=True, batch_size=32)
                    points = [models.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, row["id"])), vector=vector.tolist(), payload=row)
                        for row, vector in zip(batch, vectors)]
                    self.client.upsert(self.collection, points=points)
                    inserted.extend(point.id for point in points)
            except Exception:
                if inserted:
                    self.client.delete(self.collection, points_selector=models.PointIdsList(points=inserted))
                raise
            self.rows.extend(rows)
            self.books.append(book)
            self.refresh_lexical()
            specialists.write_library(self.identity, self.books, self.rows)
        return {**book, "chunks": len(rows), "ocr_units": len({row.get("section_index", row.get("pdf_page")) for row in rows if row.get("ocr")})}

    def remove_book(self, book_id: str) -> dict:
        from qdrant_client import models
        with self.lock:
            book = next((item for item in self.books if item["id"] == book_id), None)
            if book is None:
                raise KeyError(book_id)
            doomed = [row for row in self.rows if row["book_id"] == book_id]
            for start in range(0, len(doomed), 256):
                points = [str(uuid.uuid5(uuid.NAMESPACE_URL, row["id"])) for row in doomed[start:start+256]]
                self.client.delete(self.collection, points_selector=models.PointIdsList(points=points))
            self.rows = [row for row in self.rows if row["book_id"] != book_id]
            self.books = [item for item in self.books if item["id"] != book_id]
            self.refresh_lexical()
            specialists.write_library(self.identity, self.books, self.rows)
            if book.get("uploaded"):
                (specialists.uploads_dir(self.identity) / book["filename"]).unlink(missing_ok=True)
            return book

    def close(self):
        self.client.close()

    def search(self, queries: list[str], limit: int = 24) -> list[dict]:
        from collections import defaultdict
        from qdrant_client import models
        scores = defaultdict(float)
        compare = bool(re.search(r"\b(compar|diferen|contraste|versus|vs\b|entre .* e )", " ".join(queries), re.I))
        per_book = {}
        with self.lock:
            if not self.rows:
                return []
            for query in queries[:3]:
                vector = self.encoder.encode("query: " + query, normalize_embeddings=True)
                dense = [hit.payload["id"] for hit in self.client.query_points(self.collection, query=vector.tolist(), limit=100).points]
                lexical = self.bm25.get_scores(retrieval.lexical(query)) if self.bm25 else np.array([])
                sparse = [self.rows[i]["id"] for i in np.argsort(-lexical, kind="stable")[:100] if lexical[i] > 0]
                for ranking in (dense, sparse):
                    for rank, identity in enumerate(ranking):
                        scores[identity] += 1 / (60 + rank + 1)
                if compare and not per_book:
                    for book in self.books:
                        conditions = [models.FieldCondition(key="book_id", match=models.MatchValue(value=book["id"]))]
                        if book.get("pages", 0) > 30:
                            conditions.append(models.FieldCondition(key="pdf_page", range=models.Range(gte=6, lte=book["pages"]-2)))
                        hits = self.client.query_points(self.collection, query=vector.tolist(), limit=1,
                            query_filter=models.Filter(must=conditions)).points
                        if hits:
                            per_book[book["id"]] = hits[0].payload
            by_id = dict(self.by_id)
            page_limits = dict(self.page_limits)
        ranked = sorted(scores, key=lambda identity: (-scores[identity], identity))
        chosen = []
        if compare:
            chosen.extend(per_book.values())
            chosen = chosen[:limit]
        for identity in ranked:
            if len(chosen) >= limit:
                break
            row = by_id[identity]
            pages = page_limits[row["book_id"]]
            if pages > 30 and (row["pdf_page"] < 6 or row["pdf_page"] > pages - 2):
                continue
            if row in chosen:
                continue
            same_book = sum(x["book_id"] == row["book_id"] for x in chosen)
            if same_book >= 3 or any(unit_key(x) == unit_key(row) for x in chosen):
                continue
            chosen.append(row)
            if len(chosen) == limit:
                break
        if len(chosen) < limit:
            for identity in ranked:
                row = by_id[identity]
                if row in chosen or any(unit_key(x) == unit_key(row) for x in chosen):
                    continue
                chosen.append(row)
                if len(chosen) == limit:
                    break
        return chosen


def library(identity: str = "nei-gong") -> Library:
    if specialists.get(identity) is None:
        raise ValueError("Especialista não encontrado")
    with _library_lock:
        if identity not in _libraries:
            _libraries[identity] = Library(identity)
        return _libraries[identity]
