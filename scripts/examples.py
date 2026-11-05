"""Exemplos selecionados por regra explícita, sem substituir a avaliação agregada."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from especialistas_rag import core


def main():
    responses=core.read_jsonl(core.ROOT/'experiments/responses.jsonl')
    judgments=core.read_jsonl(core.ROOT/'experiments/judgments.jsonl')
    bykey={(r['question_id'],r['model_alias'],r['mode'],r['repetition']):r for r in responses}
    qs={q['id']:q for q in core.read_jsonl(core.ROOT/'experiments/questions.jsonl')}
    selected=[]
    for alias in core.MODELS:
        rows=[j for j in judgments if j['model_alias']==alias]
        if not rows:continue
        delta=lambda j:j['scores']['rag']['score']-j['scores']['baseline']['score']
        order=sorted(rows,key=lambda j:(delta(j),j['question_id'],j['repetition']))
        selected.append(('Maior ganho observado',order[-1]))
        selected.append(('Menor ganho observado',order[0]))
    lines=['# Exemplos auditáveis','',
           'Seleção: maior e menor diferença de notas por modelo, entre os julgamentos automáticos. Empates por ID/repetição. A seleção ilustra extremos e não representa a distribuição média. As notas são exploratórias e precisam de revisão independente.','']
    for label,j in selected:
        q=qs[j['question_id']]
        lines += [f"## {label}: {j['model_alias']} / {q['id']} / repetição {j['repetition']+1}",'',
                  q['question'],'','Critérios: '+ '; '.join(q['expected']),'',
                  'Páginas de referência: '+', '.join(q['evidence_page_ids']),'']
        for mode in ['baseline','rag']:
            r=bykey[(q['id'],j['model_alias'],mode,j['repetition'])]
            s=j['scores'][mode]
            lines += [f"### {mode}: nota automática {s['score']}/4",'',r.get('answer','[Falha de execução]'),'',
                      'Justificativa do juiz: '+s['reason'],'']
            if mode=='rag':lines+=['Fontes entregues: '+', '.join(c['id'] for c in r.get('sources',[])),'']
    (core.ROOT/'output/EXEMPLOS.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
