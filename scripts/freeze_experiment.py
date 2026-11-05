import json
import platform
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from especialistas_rag import core


def main():
    path = core.ROOT / "experiments/frozen_sources.jsonl"
    if path.exists():
        raise SystemExit("Fontes já congeladas; não sobrescrever um experimento iniciado")
    retriever = core.Retriever()
    try:
        questions = core.read_jsonl(core.ROOT / "experiments/questions.jsonl")
        sources = [{"question_id":q["id"], "sources": retriever.search(q["question"])} for q in questions]
    finally:
        retriever.close()
    core.write_jsonl(path, sources)
    records = {str(p.relative_to(core.ROOT)):core.digest(p) for p in [
        core.ROOT / "experiments/config.json", core.ROOT / "experiments/questions.jsonl", path,
        core.ROOT / "src/especialistas_rag/core.py", core.DATA / "index.json", core.ROOT / "requirements.lock.txt"]}
    manifest = {"sha256":records, "python":sys.version, "platform": platform.platform(),
                "system_prompt":core.SYSTEM,
                "retrieval_frozen":True, "no_generated_answers_used_to_build_questions":True}
    (core.ROOT / "experiments/manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("Fontes congeladas para", len(sources), "questões")


if __name__ == "__main__": main()
