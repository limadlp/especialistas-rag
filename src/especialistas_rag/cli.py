import json
from pathlib import Path
import subprocess
import sys
import httpx
import typer
from . import core

app = typer.Typer(help="RAG local e experimento Gemma/Skinner")


def request(endpoint, payload):
    try:
        r = httpx.post("http://127.0.0.1:8000" + endpoint, json=payload, timeout=600, trust_env=False)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:
        typer.echo(f"Falha na API local: {exc}. Inicie especialistas-rag serve.", err=True)
        raise typer.Exit(1)


@app.command()
def ingest():
    """Indexa o OCR local; pare a API antes de executar."""
    typer.echo(json.dumps(core.ingest(), ensure_ascii=False, indent=2))


@app.command()
def search(question: str, limit: int = 5):
    typer.echo(json.dumps(request("/search", {"question": question, "limit": limit}), ensure_ascii=False, indent=2))


@app.command()
def ask(question: str, model: str = "pequeno", rag: bool = True, json_output: bool = False):
    result = request("/answer", {"question": question, "model": model, "rag": rag})
    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        typer.echo(result["answer"])
        for source in result["sources"]:
            typer.echo(f"[{source['id']}] PDF {source['pdf_page']}, {source['side']}")


@app.command()
def chat(model: str = "pequeno", rag: bool = True):
    """Chat de teste do RAG; ainda não é o agente terapêutico completo."""
    history = []
    typer.echo("Chat local. /sair encerra; /limpar apaga o contexto desta conversa.")
    while True:
        try:
            question = typer.prompt("Você")
        except (EOFError, KeyboardInterrupt):
            return
        if question == "/sair":
            return
        if question == "/limpar":
            history = []
            continue
        result = request("/answer", {"question": question, "model": model, "rag": rag, "history": history[-8:]})
        typer.echo(result["answer"])
        history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": result["answer"]}])


@app.command()
def serve(port: int = 8000):
    import uvicorn
    uvicorn.run("especialistas_rag.api:app", host="127.0.0.1", port=port, workers=1)


@app.command("nei-gong-ingest")
def nei_gong_ingest():
    """Indexa os PDFs de Nei Gong em uma biblioteca independente."""
    from . import nei_gong
    typer.echo(json.dumps(nei_gong.ingest(), ensure_ascii=False, indent=2))


@app.command("nei-gong-serve")
def nei_gong_serve(port: int = 8001):
    """Abre a plataforma local de especialistas (alias legado)."""
    import uvicorn
    uvicorn.run("especialistas_rag.nei_gong:app", host="127.0.0.1", port=port, workers=1)


@app.command("platform-serve")
def platform_serve(port: int = 8001):
    """Abre a plataforma local de especialistas e bibliotecas."""
    import uvicorn
    uvicorn.run("especialistas_rag.nei_gong:app", host="127.0.0.1", port=port, workers=1)


@app.command()
def evaluate(repetitions: int = 3):
    """Executa o benchmark retomável congelado, com LM Studio em execução."""
    subprocess.run([sys.executable, str(core.ROOT / "scripts/run_experiment.py"),
                    "--repetitions", str(repetitions)], check=True)


@app.command()
def report():
    subprocess.run([sys.executable, str(core.ROOT / "scripts/report.py")], check=True)


@app.command()
def status():
    """Mostra progresso do experimento, sem interferir na inferência."""
    subprocess.run([sys.executable, str(core.ROOT / "scripts/progress.py")], check=True)


if __name__ == "__main__":
    app()
