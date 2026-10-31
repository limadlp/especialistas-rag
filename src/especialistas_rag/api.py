from contextlib import asynccontextmanager
from threading import Lock
import os
import time
import httpx

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from . import core


class Query(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=5, ge=1, le=10)


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class AnswerQuery(Query):
    model: str = "pequeno"
    rag: bool = True
    history: list[Message] = Field(default_factory=list, max_length=8)


@asynccontextmanager
async def lifespan(app):
    app.state.retriever = None
    app.state.lock = Lock()
    app.state.generation_lock = Lock()
    yield
    if app.state.retriever:
        app.state.retriever.close()


app = FastAPI(title="Especialistas — experimento Skinner", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("TAC_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


def retriever():
    with app.state.lock:
        if app.state.retriever is None:
            app.state.retriever = core.Retriever()
    return app.state.retriever


@app.get("/health")
def health():
    try:
        response = httpx.get(core.LM_URL + "/api/v1/models", timeout=2, trust_env=False)
        reachable = response.is_success
    except httpx.HTTPError:
        reachable = False
    return {"index_exists": (core.DATA / "index.json").exists(), "lm_url": core.LM_URL,
            "lmstudio_reachable": reachable, "thinking": False, "models": core.MODELS}


@app.post("/search")
def search(query: Query):
    try:
        return {"sources": retriever().search(query.question, query.limit)}
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/answer")
def answer(query: AnswerQuery):
    if query.model not in core.MODELS and query.model not in core.MODELS.values():
        raise HTTPException(422, "Escolha um dos três Gemma configurados")
    try:
        with app.state.generation_lock:
            start = time.perf_counter()
            sources = retriever().search(query.question, query.limit) if query.rag else []
            retrieval_seconds = time.perf_counter() - start
            result = core.answer(query.question, query.model, sources, [m.model_dump() for m in query.history])
        result.pop("raw", None)
        return {**result, "retrieval_seconds": retrieval_seconds}
    except Exception as exc:
        raise HTTPException(503, f"Falha na consulta local: {exc}") from exc
