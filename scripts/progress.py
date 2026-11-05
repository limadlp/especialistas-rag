"""Estado compacto, sem abrir o Qdrant ou consultar o modelo."""
import json
from pathlib import Path
import sys
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from especialistas_rag import core


def complete_rows(path):
    if not path.exists():return []
    content=path.read_text()
    lines=content.splitlines()
    if content and not content.endswith('\n'):lines=lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def status():
    responses=complete_rows(core.ROOT/'experiments/responses.jsonl')
    judgments=complete_rows(core.ROOT/'experiments/judgments.jsonl')
    failures=complete_rows(core.ROOT/'build/judge-errors.jsonl')
    log=core.ROOT/'build/experiment.log'
    complete=log.exists() and log.read_text().rstrip().endswith('PIPELINE CONCLUÍDO')
    return {'phase':'complete' if complete else ('judging' if len(responses)==648 else 'generation'),
            'responses':len(responses),'planned_responses':648,'judged_pairs':len(judgments),
            'planned_pairs':324,'errors':sum(r['status']!='ok' for r in responses),
            'judgment_processing_errors':len(failures),
            'judgment_pairs_with_processing_error':len({tuple(r['key']) for r in failures}),
            'reasoning_tokens':sum(r.get('stats',{}).get('reasoning_output_tokens',0) for r in responses),
            'by_model':dict(Counter(r['model_alias'] for r in responses))}


if __name__=='__main__':print(json.dumps(status(),ensure_ascii=False,indent=2))
