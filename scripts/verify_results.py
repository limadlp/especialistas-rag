"""Verifica integridade e aritmética; não valida a correção clínica das notas."""
import collections
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from especialistas_rag import core


def main():
    rs=core.read_jsonl(core.ROOT/'experiments/responses.jsonl')
    js=core.read_jsonl(core.ROOT/'experiments/judgments.jsonl')
    summary=json.loads((core.ROOT/'output/summary.json').read_text())
    manifest=json.loads((core.ROOT/'experiments/manifest.json').read_text())
    frozen={r['question_id']:r['sources'] for r in core.read_jsonl(core.ROOT/'experiments/frozen_sources.jsonl')}
    keys=[(r['question_id'],r['model_alias'],r['mode'],r['repetition']) for r in rs]
    jkeys=[(j['question_id'],j['model_alias'],j['repetition']) for j in js]
    checks={
        '648_unique_runs':len(keys)==len(set(keys))==648,
        '324_unique_pairs':len(jkeys)==len(set(jkeys))==324,
        'balanced_108_per_condition':all(sum(r['model_alias']==m and r['mode']==mode for r in rs)==108 for m in core.MODELS for mode in ['baseline','rag']),
        'all_question_repetitions_complete':all(collections.Counter(r['question_id'] for r in rs if r['model_alias']==m and r['mode']==mode)=={f'test-{i:02}':3 for i in range(1,37)} for m in core.MODELS for mode in ['baseline','rag']),
        'frozen_files_unchanged':all(core.digest(core.ROOT/p)==h for p,h in manifest['sha256'].items()),
        'identical_retrieved_sources':all(r.get('sources',[])==(frozen[r['question_id']] if r['mode']=='rag' else []) for r in rs if r['status']=='ok'),
        'zero_reported_reasoning_tokens':all(r.get('stats',{}).get('reasoning_output_tokens')==0 for r in rs if r['status']=='ok'),
        'known_model_identifiers':all(r['model_key']==core.MODELS[r['model_alias']] for r in rs),
        'returned_model_matches_request':all(r.get('model')==core.MODELS[r['model_alias']] for r in rs if r['status']=='ok'),
        'blinding_maps_valid':all(set(j['blinding'])=={'A','B'} and set(j['blinding'].values())=={'baseline','rag'} for j in js),
    }
    for m in core.MODELS:
        bymode={mode:[j['scores'][mode]['score'] for j in js if j['model_alias']==m] for mode in ['baseline','rag']}
        if all(bymode.values()):
            direct_gain=(statistics.mean(bymode['rag'])-statistics.mean(bymode['baseline']))*25
            reported=next(g['gain_points'] for g in summary['gains'] if g['model']==m)
            checks[f'gain_arithmetic_{m}']=abs(direct_gain-reported)<1e-9
            for mode in bymode:
                reported=next(c['score_100'] for c in summary['conditions'] if c['model']==m and c['mode']==mode)
                checks[f'score_arithmetic_{m}_{mode}']=abs(statistics.mean(bymode[mode])*25-reported)<1e-9
    result={'structural_checks':checks,'all_structural_checks_passed':all(checks.values()),
            'generation_failures':sum(r['status']!='ok' for r in rs),
            'possibly_truncated':sum(r.get('possibly_truncated',False) for r in rs),
            'domain_accuracy_validated':False,'expert_review':'not_performed','codex_review_responses':48}
    (core.ROOT/'output/integrity.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':main()
