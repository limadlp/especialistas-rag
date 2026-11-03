"""Local specialist registry and library metadata."""
import json
from pathlib import Path
import re
import threading
import unicodedata
import uuid

from .. import settings
from ..files import write_jsonl

REGISTRY = settings.DATA / "specialists.json"
DATA_ROOT = settings.DATA / "specialists"
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
        return settings.DATA / "nei_gong"
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
    write_jsonl(folder / "chunks.jsonl", rows)
    (folder / "books.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "index.json").write_text(json.dumps({"books": len(catalog), "chunks": len(rows)}, indent=2), encoding="utf-8")
