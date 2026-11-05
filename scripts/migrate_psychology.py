"""Copy the existing Skinner index into the separate psychology specialist."""
from pathlib import Path
import sys
import uuid

import numpy as np
from pypdf import PdfReader
from qdrant_client import QdrantClient, models

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from especialistas_rag import core
from especialistas_rag.adapters import specialists


def main():
    source = core.DATA
    target = specialists.data_dir("psychology")
    if (target / "books.json").exists():
        print("Psychology index already initialized; leaving it unchanged.")
        return
    rows = core.read_jsonl(source / "chunks.jsonl")
    vectors = np.load(source / "vectors.npy", mmap_mode="r")
    if len(rows) != len(vectors):
        raise RuntimeError("Existing chunks and vectors differ in length")
    pdf = next((core.ROOT / "source_books" / "behavior").glob("skinner - *.pdf"))
    reader = PdfReader(pdf)
    book = {"id": "skinner-ciencia-e-comportamento-humano", "title": "Ciência e comportamento humano",
            "author": "B. F. Skinner", "filename": pdf.name, "pages": len(reader.pages),
            "sha256": core.digest(pdf), "ocr_language": "por", "uploaded": False}
    migrated = [{"id": f"{book['id']}-{row['id']}", "book_id": book["id"], "title": book["title"],
                 "author": book["author"], "pdf_page": row["pdf_page"], "text": row["text"],
                 "ocr": True} for row in rows]
    target.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(target / "qdrant"))
    try:
        if not client.collection_exists("books"):
            client.create_collection("books", vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
        for start in range(0, len(migrated), 128):
            batch = migrated[start:start+128]
            client.upsert("books", points=[models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, row["id"])), vector=vector.tolist(), payload=row)
                for row, vector in zip(batch, vectors[start:start+128])])
    finally:
        client.close()
    specialists.write_library("psychology", [book], migrated)
    print(f"Psychology initialized: {len(migrated)} Skinner excerpts")


if __name__ == "__main__":
    main()
