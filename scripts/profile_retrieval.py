"""Medição auxiliar após geração; sem ajustar índice ou fontes congeladas."""
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from especialistas_rag import core


def main():
    qs=core.read_jsonl(core.ROOT/'experiments/questions.jsonl')
    frozen={r['question_id']:r['sources'] for r in core.read_jsonl(core.ROOT/'experiments/frozen_sources.jsonl')}
    start=time.perf_counter();r=core.Retriever();initialization=time.perf_counter()-start
    try:
        r.search(next(q['question'] for q in qs if q['id']=='dev-01'))
        rows=[]
        for q in qs:
            if q['split']!='test':continue
            start=time.perf_counter();hits=r.search(q['question']);elapsed=time.perf_counter()-start
            rows.append({'question_id':q['id'],'seconds':elapsed,
                         'same_ids_as_frozen':[h['id'] for h in hits]==[h['id'] for h in frozen[q['id']]]})
        q=next(q for q in qs if q['id']=='test-01')
        vec=r.encoder.encode('query: '+q['question'],normalize_embeddings=True)
        dense=r.client.query_points('skinner',query=vec.tolist(),limit=50).points
        sparse_scores=r.bm25.get_scores(core.lexical(q['question']))
        sparse_ids=np.argsort(-sparse_scores,kind='stable')[:50]
        diagnostic={'question_id':'test-01','expected_page':'pdf051-left',
                    'dense_reference_ranks_at_50':[i+1 for i,h in enumerate(dense) if h.payload['page_id']=='pdf051-left'],
                    'lexical_reference_ranks_at_50':[i+1 for i,j in enumerate(sparse_ids) if r.chunks[j]['page_id']=='pdf051-left'],
                    'frozen_final_ids':[h['id'] for h in frozen['test-01']]}
    finally:r.close()
    result={'environment_note':'Medição auxiliar em CPU após a geração principal, com o juiz local em execução. Não é latência ponta a ponta da API.',
            'initialization_seconds':initialization,'questions':len(rows),
            'median_seconds':float(np.median([x['seconds'] for x in rows])),
            'p95_seconds':float(np.percentile([x['seconds'] for x in rows],95)),
            'all_sources_reproduced':all(x['same_ids_as_frozen'] for x in rows),
            'samples':rows,'diagnostic':diagnostic}
    (core.ROOT/'output/retrieval_profile.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))


if __name__=='__main__':main()
