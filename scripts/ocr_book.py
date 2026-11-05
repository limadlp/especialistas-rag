"""OCR retomável; cada face mantém sua localização no PDF original."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def process_page(args):
    path, i, dpi = args
    import pypdfium2 as pdfium
    output = ROOT / "data/ocr/pages" / f"{i + 1:03}.json"
    if output.exists():
        return i + 1
    doc = pdfium.PdfDocument(path)
    page = doc[i]
    image = page.render(scale=dpi / 72).to_pil().convert("RGB")
    # Source is a spread; preserve the mapping rather than invent page numbers.
    regions = [("left", (0, 0, image.width // 2, image.height)),
               ("right", (image.width // 2, 0, image.width, image.height))]
    if image.width / image.height < 1.1:
        regions = [("single", (0, 0, image.width, image.height))]
    records = []
    for side, box in regions:
        crop = image.crop(box)
        with tempfile.TemporaryDirectory(prefix="tac-ocr-") as temp:
            png = Path(temp) / "page.png"
            crop.save(png)
            env = dict(os.environ, OMP_THREAD_LIMIT="1")
            result = subprocess.run(
                ["tesseract", str(png), "stdout", "--tessdata-dir",
                 str(ROOT / "data/tessdata"), "-l", "por", "--psm", "3"],
                capture_output=True, text=True, check=True, env=env,
            )
        raw = result.stdout.strip()
        first = raw.splitlines()[:4]
        printed = next((int(s.strip()) for s in first if re.fullmatch(r"\d{1,3}", s.strip())), None)
        text = re.sub(r"(\w)[\-\u00ad]\s*\n(\w)", r"\1\2", raw)
        text = text.replace("\u00ad", "")
        records.append({"id": f"pdf{i+1:03}-{side}", "pdf_page": i + 1,
                        "side": side, "printed_page_candidate": printed,
                        "printed_page_verified": False, "text": text,
                        "raw_text": raw})
        if i in [4, 19, 49, 99, 149, 199, 249]:
            crop.thumbnail((900, 1300))
            crop.save(ROOT / "data/ocr/samples" / f"{i+1:03}-{side}.png")
    output.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return i + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=230)
    args = parser.parse_args()
    from pypdf import PdfReader
    path = next((ROOT / "source_books").glob("skinner*.pdf"))
    count = len(PdfReader(path).pages)
    for folder in ["pages", "samples"]:
        (ROOT / "data/ocr" / folder).mkdir(parents=True, exist_ok=True)
    tasks = [(str(path), i, args.dpi) for i in range(count)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, i in enumerate(pool.map(process_page, tasks), 1):
            if n % 10 == 0 or n == count:
                print(f"OCR {n}/{count} páginas PDF", flush=True)
    pages = []
    for i in range(count):
        pages.extend(json.loads((ROOT / "data/ocr/pages" / f"{i+1:03}.json").read_text()))
    (ROOT / "data/ocr/book.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pages), encoding="utf-8")
    manifest = {"source": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "pdf_pages": count, "faces": len(pages), "dpi": args.dpi,
                "ocr": "tesseract por psm3", "nonempty_faces": sum(len(p['text']) > 100 for p in pages)}
    (ROOT / "data/ocr/manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
