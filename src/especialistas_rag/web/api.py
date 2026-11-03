"""FastAPI routes and browser assets for the specialist platform."""
import json
import sqlite3
import threading
import uuid
from pathlib import Path
from urllib.parse import unquote

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import rag, settings
from ..adapters import conversations as conversation_store
from ..adapters import documents, llm, specialists as registry
from ..adapters.library import library

DATA = settings.DATA / "nei_gong"
CHAT_DB = settings.NEI_GONG_CHAT_DB
BOOKS = settings.SOURCE_BOOKS / "nei gong"
WEB = Path(__file__).parent / "static"


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=12000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    model: str | None = None
    specialist_id: str = "nei-gong"
    thinking: bool = False
    conversation_id: str | None = Field(default=None, max_length=64)
    history: list[Message] = Field(default_factory=list, max_length=12)


class SpecialistRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=1000)
    ocr_language: str = "por"


class PromptRequest(BaseModel):
    system_prompt: str | None = Field(default=None, max_length=12000)


app = FastAPI(title="Especialistas — biblioteca local")
_chat_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def checked_specialist(identity: str) -> dict:
    specialist = registry.get(identity)
    if specialist is None:
        raise HTTPException(404, "Especialista não encontrado")
    return specialist


def conversations_for(identity: str):
    checked_specialist(identity)
    return {"conversations": conversation_store.list_conversations(CHAT_DB if identity == "nei-gong" else registry.chat_db(identity))}


def conversation_for(identity: str, conversation_id: str):
    checked_specialist(identity)
    record = conversation_store.get_conversation(CHAT_DB if identity == "nei-gong" else registry.chat_db(identity), conversation_id)
    if record is None:
        raise HTTPException(404, "Conversa não encontrada")
    return record


@app.get("/")
def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/app.css")
def css():
    return FileResponse(WEB / "app.css", media_type="text/css", headers={"Cache-Control": "no-store"})


@app.get("/app.js")
def js():
    return FileResponse(WEB / "app.js", media_type="text/javascript", headers={"Cache-Control": "no-store"})


@app.get("/markdown.js")
def markdown_js():
    return FileResponse(WEB / "markdown.js", media_type="text/javascript", headers={"Cache-Control": "no-store"})


@app.get("/api/status")
def status():
    return {"indexed": (DATA / "index.json").exists(), "books": json.loads((DATA / "books.json").read_text()) if (DATA / "books.json").exists() else [],
            "source_books": len(list(BOOKS.glob("*.pdf")))}


@app.get("/api/specialists")
def specialists():
    return {"specialists": [{**entry, "book_count": len(registry.books(entry["id"]))}
                            for entry in registry.read_registry()]}


@app.post("/api/specialists", status_code=201)
def create_specialist(request: SpecialistRequest):
    try:
        return registry.create(request.name, request.description, request.ocr_language)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/specialists/{identity}/books")
def specialist_books(identity: str):
    checked_specialist(identity)
    return {"books": registry.books(identity)}


@app.get("/api/specialists/{identity}/prompt")
def get_prompt(identity: str):
    specialist = checked_specialist(identity)
    return {"system_prompt": rag.system_prompt(identity), "custom": bool(specialist.get("system_prompt"))}


@app.put("/api/specialists/{identity}/prompt")
def put_prompt(identity: str, request: PromptRequest):
    checked_specialist(identity)
    try:
        registry.set_system_prompt(identity, request.system_prompt)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return get_prompt(identity)


@app.post("/api/specialists/{identity}/books", status_code=202)
async def upload_book(identity: str, request: Request):
    specialist = checked_specialist(identity)
    language = request.headers.get("x-ocr-language", specialist["ocr_language"])
    if language not in ("por", "eng"):
        raise HTTPException(422, "Idioma OCR inválido")
    password = unquote(request.headers.get("x-pdf-password", ""))
    filename = Path(unquote(request.headers.get("x-filename", "livro.pdf"))).name
    extension = Path(filename).suffix.lower()
    if extension not in (".pdf", ".epub"):
        raise HTTPException(422, "Envie um arquivo .pdf ou .epub")
    content_type = request.headers.get("content-type", "").split(";")[0]
    expected_type = "application/epub+zip" if extension == ".epub" else "application/pdf"
    if content_type != expected_type:
        raise HTTPException(415, f"Tipo de arquivo inválido; esperado {expected_type}")
    folder = registry.uploads_dir(identity)
    folder.mkdir(parents=True, exist_ok=True)
    temp = folder / f"{uuid.uuid4().hex}{extension}"
    size = 0
    try:
        with temp.open("wb") as output:
            async for chunk in request.stream():
                size += len(chunk)
                if size > 200 * 1024 * 1024:
                    raise HTTPException(413, "Livro maior que o limite de 200 MB")
                output.write(chunk)
        with temp.open("rb") as uploaded:
            signature = uploaded.read(5)
        if size < 8 or (extension == ".pdf" and signature != b"%PDF-") or (extension == ".epub" and signature[:2] != b"PK"):
            raise HTTPException(422, "Arquivo inválido para o formato informado")
        book = documents.book_metadata(temp, filename, language, password)
        if any(item["sha256"] == book["sha256"] for item in registry.books(identity)):
            raise HTTPException(409, "Este livro já está na biblioteca deste especialista")
    except Exception as exc:
        temp.unlink(missing_ok=True)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(422, f"Não foi possível ler o livro: {exc}") from exc
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"id": job_id, "status": "queued", "stage": "Aguardando indexação", "current": 0,
                         "total": book.get("pages", book.get("sections", 0))}

    def progress(current, total):
        with _jobs_lock:
            _jobs[job_id].update(status="running", stage="Extraindo texto e aplicando OCR quando necessário", current=current, total=total)

    def worker():
        try:
            result = library(identity).add_book(temp, book, progress, password)
            with _jobs_lock:
                _jobs[job_id].update(status="done", stage="Livro pronto", result=result,
                                     current=book.get("pages", book.get("sections", 0)))
        except Exception as exc:
            temp.unlink(missing_ok=True)
            with _jobs_lock:
                _jobs[job_id].update(status="failed", stage="Falha na indexação", error=str(exc))

    threading.Thread(target=worker, daemon=True, name=f"book-{job_id[:8]}").start()
    return {"job_id": job_id, "book": book}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Tarefa não encontrada")
        return dict(job)


@app.delete("/api/specialists/{identity}/books/{book_id}")
def delete_book(identity: str, book_id: str):
    checked_specialist(identity)
    try:
        return {"book": library(identity).remove_book(book_id)}
    except KeyError as exc:
        raise HTTPException(404, "Livro não encontrado") from exc


@app.get("/api/models")
def list_models():
    try:
        available = llm.models()
        default = llm.default_model(available)
        return {"models": available, "default_model": default["id"] if default else None}
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/api/conversations")
def conversations():
    return conversations_for("nei-gong")


@app.get("/api/conversations/{identity}")
def conversation(identity: str):
    return conversation_for("nei-gong", identity)


@app.get("/api/specialists/{identity}/conversations")
def specialist_conversations(identity: str):
    return conversations_for(identity)


@app.get("/api/specialists/{identity}/conversations/{conversation_id}")
def specialist_conversation(identity: str, conversation_id: str):
    return conversation_for(identity, conversation_id)


@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        checked_specialist(request.specialist_id)
        available = llm.models()
        model = next((entry for entry in available if entry["id"] == request.model), None)
        if request.model is None:
            model = llm.default_model(available)
        if model is None:
            raise HTTPException(422, "Carregue um modelo no LM Studio ou selecione um modelo disponível.")
        model_id = model["id"]
        chat_db = CHAT_DB if request.specialist_id == "nei-gong" else registry.chat_db(request.specialist_id)
        with _chat_lock:
            if request.conversation_id:
                record = conversation_store.get_conversation(chat_db, request.conversation_id)
                if record is None:
                    raise HTTPException(404, "Conversa não encontrada")
                history = [{"role": x["role"], "content": x["content"]} for x in record["messages"][-8:]]
            else:
                history = [x.model_dump() for x in request.history]
            selected, retrieval = rag.retrieve(request.question, model_id, history, request.specialist_id)
            sources = [{**source, "citation": f"S{i}"} for i, source in enumerate(selected, 1)]
            answer = rag.complete(request.question, model, sources, history, request.specialist_id, request.thinking)
            identity = conversation_store.save_turn(
                chat_db, request.conversation_id, request.question, answer, model_id, sources, retrieval,
                prior=history if not request.conversation_id else None,
            )
            return {"answer": answer, "sources": sources, "retrieval": retrieval,
                    "conversation_id": identity}
    except HTTPException:
        raise
    except (RuntimeError, OSError, ValueError, sqlite3.Error, httpx.HTTPError) as exc:
        raise HTTPException(503, str(exc)) from exc
