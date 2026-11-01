"""Local specialist registry and incremental PDF/EPUB indexing."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import unicodedata
import uuid

from pypdf import PdfReader

from . import core
from . import epub

ROOT = core.ROOT
REGISTRY = ROOT / "data" / "specialists.json"
DATA_ROOT = ROOT / "data" / "specialists"
_registry_lock = threading.Lock()


def slug(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")[:50]


def defaults() -> list[dict]:
    return [
        {"id": "nei-gong", "name": "Nei Gong", "description": "Nei Gong, alquimia interna daoista e práticas contemplativas correlatas", "ocr_language": "eng"},
        {"id": "psychology", "name": "Psicologia", "description": "Análise do comportamento e psicologia baseada nos livros da biblioteca", "ocr_language": "por"},
    ]


def read_registry() -> list[dict]:
    with _registry_lock:
        if not REGISTRY.exists():
            REGISTRY.parent.mkdir(parents=True, exist_ok=True)
            REGISTRY.write_text(json.dumps(defaults(), ensure_ascii=False, indent=2), encoding="utf-8")
        return json.loads(REGISTRY.read_text(encoding="utf-8"))


def get(identity: str) -> dict | None:
    return next((item for item in read_registry() if item["id"] == identity), None)


def set_system_prompt(identity: str, prompt: str | None) -> dict:
    with _registry_lock:
        entries = json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.exists() else defaults()
        entry = next((item for item in entries if item["id"] == identity), None)
        if entry is None:
            raise KeyError(identity)
        if prompt is None:
            entry.pop("system_prompt", None)
        else:
            value = prompt.strip()
            if not value or len(value) > 12000:
                raise ValueError("O system prompt deve ter entre 1 e 12000 caracteres")
            entry["system_prompt"] = value
        temp = REGISTRY.with_suffix(".tmp")
        temp.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(REGISTRY)
        return entry


def create(name: str, description: str, ocr_language: str = "por") -> dict:
    name = name.strip()
    if not name or len(name) > 80:
        raise ValueError("Nome deve ter entre 1 e 80 caracteres")
    if ocr_language not in ("por", "eng"):
        raise ValueError("Idioma OCR inválido")
    with _registry_lock:
        entries = json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.exists() else defaults()
        base = slug(name) or "especialista"
        identity = base
        while any(item["id"] == identity for item in entries):
            identity = f"{base}-{uuid.uuid4().hex[:6]}"
        entry = {"id": identity, "name": name, "description": description.strip()[:1000], "ocr_language": ocr_language}
        entries.append(entry)
        REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        temp = REGISTRY.with_suffix(".tmp")
        temp.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(REGISTRY)
    data_dir(identity).mkdir(parents=True, exist_ok=True)
    return entry


def data_dir(identity: str) -> Path:
    if identity == "nei-gong":
        return ROOT / "data" / "nei_gong"
    return DATA_ROOT / identity


def uploads_dir(identity: str) -> Path:
    return data_dir(identity) / "uploads"


def chat_db(identity: str) -> Path:
    return data_dir(identity) / "conversations.sqlite3"


def books(identity: str) -> list[dict]:
    path = data_dir(identity) / "books.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def write_library(identity: str, catalog: list[dict], rows: list[dict]):
    folder = data_dir(identity)
    folder.mkdir(parents=True, exist_ok=True)
    core.write_jsonl(folder / "chunks.jsonl", rows)
    (folder / "books.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "index.json").write_text(json.dumps({"books": len(catalog), "chunks": len(rows)}, indent=2), encoding="utf-8")


def ocr_image(image, language: str) -> str:
    with tempfile.TemporaryDirectory(prefix="specialist-ocr-") as folder:
        image_path = Path(folder) / "page.png"
        image.save(image_path)
        command = ["tesseract", str(image_path), "stdout", "-l", language, "--psm", "3"]
        if language == "por":
            command[3:3] = ["--tessdata-dir", str(ROOT / "data" / "tessdata")]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout.strip()


def extract_pages(pdf: Path, language: str, progress=None, password: str = "") -> tuple[list[dict], int]:
    """Extract text per PDF page, using OCR only when embedded text is absent."""
    import pypdfium2 as pdfium
    reader = open_pdf(pdf, password)
    page_count = len(reader.pages)
    if page_count == 0:
        raise ValueError("PDF sem páginas")
    document = None
    pages = []
    for index, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        used_ocr = False
        if len(re.sub(r"\W", "", text)) < 80:
            if document is None:
                document = pdfium.PdfDocument(str(pdf), password=password or None)
            image = document[index-1].render(scale=200/72).to_pil().convert("RGB")
            text = ocr_image(image, language)
            used_ocr = True
        pages.append({"pdf_page": index, "text": text, "ocr": used_ocr})
        if progress:
            progress(index, page_count)
    return pages, page_count


def chunks_for_book(path: Path, book: dict, encoder, progress=None, password: str = "") -> list[dict]:
    is_epub = book.get("format") == "epub" or path.suffix.lower() == ".epub"
    units = (epub.extract(path, book["ocr_language"], ocr_image, progress) if is_epub
             else extract_pages(path, book["ocr_language"], progress, password)[0])
    rows = []
    for unit in units:
        text = re.sub(r"[ \t]+", " ", unit["text"]).strip()
        if len(text) < 80:
            continue
        offsets = encoder.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        for number, start in enumerate(range(0, len(offsets), 250)):
            stop = min(start + 320, len(offsets))
            chunk = text[offsets[start][0]:offsets[stop-1][1]].strip()
            if len(chunk) >= 80:
                if is_epub:
                    location = {"section_index": unit["section_index"], "section_title": unit["section_title"],
                                "chunk_index": number}
                    row_id = f"{book['id']}-s{unit['section_index']:04}-c{number:03}"
                else:
                    location = {"pdf_page": unit["pdf_page"]}
                    row_id = f"{book['id']}-p{unit['pdf_page']:04}-c{number:02}"
                rows.append({"id": row_id, "book_id": book["id"], "title": book["title"],
                             "author": book["author"], "text": chunk, "ocr": unit["ocr"], **location})
            if stop == len(offsets):
                break
    if not rows:
        raise ValueError("Não foi possível extrair texto útil deste livro. EPUBs protegidos por DRM não são suportados.")
    return rows


def open_pdf(path: Path, password: str = ""):
    reader = PdfReader(path)
    if reader.is_encrypted:
        try:
            if not reader.decrypt(password):
                raise ValueError("Senha do PDF incorreta ou não informada")
        except ImportError as exc:
            raise ValueError("Este PDF precisa do pacote cryptography para ser aberto") from exc
    return reader


def book_metadata(path: Path, original_name: str, language: str, password: str = "") -> dict:
    is_epub = path.suffix.lower() == ".epub"
    if is_epub:
        meta, entries = epub.package(path)
        title = meta["title"] or Path(original_name).stem
        author = meta["author"] or "Autor não identificado"
        count = {"sections": len(entries)}
    else:
        reader = open_pdf(path, password)
        meta = reader.metadata or {}
        title = str(meta.get("/Title") or Path(original_name).stem).strip() or Path(original_name).stem
        author = str(meta.get("/Author") or "").strip() or "Autor não identificado"
        count = {"pages": len(reader.pages)}
    with path.open("rb") as uploaded:
        digest = hashlib.file_digest(uploaded, "sha256").hexdigest()
    return {"id": uuid.uuid4().hex, "title": title, "author": author, "filename": path.name,
            "original_filename": original_name, "format": "epub" if is_epub else "pdf", **count,
            "sha256": digest,
            "ocr_language": language, "uploaded": True}
