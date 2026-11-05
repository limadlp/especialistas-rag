"""Benchmark retomável 3×2. Uma linha por execução, inclusive falhas."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import httpx
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from especialistas_rag import core
from especialistas_rag.adapters.lmstudio import headers as lmstudio_headers

LMS = str(Path.home() / ".lmstudio/bin/lms")


def command(args):
    result = subprocess.run([LMS, *args], capture_output=True, text=True, timeout=300)
    with (core.ROOT / "build/lmstudio-load.log").open("a") as f:
        f.write(result.stdout + result.stderr + "\n")
    if result.returncode:
        raise RuntimeError(f"lms falhou ({result.returncode}); veja build/lmstudio-load.log")


def load_model(model):
    inventory = httpx.get(core.LM_URL + "/api/v1/models", headers=lmstudio_headers(), trust_env=False).json()["models"]
    for item in inventory:
        for instance in item.get("loaded_instances", []):
            # Leave unrelated user models alone. Stop to avoid memory pressure.
            identity = instance.get("id", instance.get("instance_id", ""))
            known = item.get("key") in core.MODELS.values() or identity.startswith("tac-")
            if not known:
                raise RuntimeError(f"Outro modelo carregado ({item['key']}); liberar antes do benchmark")
            if identity:
                command(["unload", identity])
    command(["load", model["key"], "--identifier", model["key"], "--context-length", "8192", "--yes"])


def verify_manifest():
    manifest = json.loads((core.ROOT / "experiments/manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if core.digest(core.ROOT / name) != expected:
            raise RuntimeError(f"Arquivo congelado alterado: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", choices=list(core.MODELS))
    args = parser.parse_args()
    verify_manifest()
    config = json.loads((core.ROOT / "experiments/config.json").read_text())
    models = [m for m in config["models"] if not args.model or m["alias"] == args.model]
    questions = [q for q in core.read_jsonl(core.ROOT / "experiments/questions.jsonl") if q["split"] == args.split]
    if args.limit: questions = questions[:args.limit]
    frozen = {q["question_id"]:q["sources"] for q in core.read_jsonl(core.ROOT / "experiments/frozen_sources.jsonl")}
    destination = core.ROOT / "experiments" / ("responses.jsonl" if args.split == "test" else "dev_responses.jsonl")
    done = set()
    if destination.exists():
        for row in core.read_jsonl(destination):
            done.add((row["question_id"], row["model_alias"], row["mode"], row["repetition"]))
    for repetition in range(args.repetitions):
        rotated = models[repetition % len(models):] + models[:repetition % len(models)]
        for model in rotated:
            order = list(questions)
            random.Random(config["question_order_seed"] + repetition).shuffle(order)
            jobs = [(q, mode) for q in order for mode in (["rag", "baseline"] if random.Random(q["id"]+str(repetition)).random() > .5 else ["baseline", "rag"])]
            if all((q["id"], model["alias"], mode, repetition) in done for q,mode in jobs):
                continue
            print(f"CARREGANDO {model['label']} rep={repetition+1}", flush=True)
            load_model(model)
            warmup = core.answer("Responda apenas: pronto.", model["key"], max_tokens=16)
            with (core.ROOT / "experiments/warmup.jsonl").open("a") as f:
                f.write(json.dumps({"model":model["key"], "repetition":repetition, **warmup}, ensure_ascii=False)+"\n")
            def execute(q, mode):
                identity = (q["id"], model["alias"], mode, repetition)
                row = {"question_id":q["id"], "category":q["category"], "question":q["question"],
                       "model_alias":model["alias"], "model_key":model["key"], "mode":mode,
                       "repetition":repetition, "timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat()}
                try:
                    result = core.answer(q["question"], model["key"], frozen[q["id"]] if mode == "rag" else [])
                    row.update(result)
                    row["status"] = "ok"
                except Exception as exc:
                    row.update(status="error", error=str(exc))
                row['request_concurrency'] = 1 if args.split=='dev' else config['request_concurrency']
                return identity, row
            concurrency = 1 if args.split=='dev' else config['request_concurrency']
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(execute,q,mode) for q,mode in jobs
                           if (q['id'],model['alias'],mode,repetition) not in done]
                for future in as_completed(futures):
                    identity,row = future.result()
                    with destination.open("a") as f:
                        f.write(json.dumps(row, ensure_ascii=False)+"\n")
                        f.flush()
                    done.add(identity)
                    print(f"{len(done)} {identity} {row['status']} {row.get('generation_seconds',0):.1f}s", flush=True)
    print("EXECUÇÕES CONCLUÍDAS", str(destination), flush=True)


if __name__ == "__main__": main()
