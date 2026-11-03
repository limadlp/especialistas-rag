"""Read PDF and EPUB input, with Tesseract as an OCR fallback."""
import hashlib
from pathlib import Path
import re
import subprocess
import tempfile
import uuid

from pypdf import PdfReader

from .. import epub, settings


def ocr_image(image, language: str) -> str:
    with tempfile.TemporaryDirectory(prefix="specialist-ocr-") as folder:
        image_path = Path(folder) / "page.png"
        image.save(image_path)
        command = ["tesseract", str(image_path), "stdout", "-l", language, "--psm", "3"]
        local_language = settings.DATA / "tessdata" / f"{language}.traineddata"
        if local_language.is_file():
            command[3:3] = ["--tessdata-dir", str(local_language.parent)]
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
