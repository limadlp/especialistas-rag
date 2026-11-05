"""Pacote de revisão humana; separa respostas cegas da chave de identificação."""
import csv
import json
from pathlib import Path
import random
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from especialistas_rag import core


def main():
    rows=core.read_jsonl(core.ROOT/'experiments/responses.jsonl')
    questions={q['id']:q for q in core.read_jsonl(core.ROOT/'experiments/questions.jsonl')}
    random.Random(854821).shuffle(rows)
    output=core.ROOT/'output/review'
    output.mkdir(parents=True,exist_ok=True)
    existing=output/'blinded.csv'
    if existing.exists():
        with existing.open() as f:
            if any(r.get('score_0_4','').strip() for r in csv.DictReader(f)):
                raise SystemExit('Há notas humanas preenchidas; preserve o arquivo antes de gerar outro pacote.')
    packet=[];key=[]
    for number,r in enumerate(rows,1):
        identity=f'R{number:04}'
        q=questions[r['question_id']]
        clean=re.sub(r'\[pdf\d{3}-(?:left|right|single)-c\d+\]','',r.get('answer','[SEM RESPOSTA]'))
        packet.append({'review_id':identity,'question':q['question'],'answer':clean,
                       'expected':'; '.join(q['expected']),'evidence_page_ids':'; '.join(q['evidence_page_ids']),
                       'score_0_4':'','major_error':'','comment':''})
        key.append({'review_id':identity,'question_id':r['question_id'],'model_alias':r['model_alias'],
                    'mode':r['mode'],'repetition':r['repetition'],'status':r['status']})
    for name,rs in [('blinded.csv',packet),('key.csv',key)]:
        with (output/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    (output/'LEIA-ME.md').write_text(
        '# Revisão humana\n\nAvalie blinded.csv sem abrir key.csv ou as notas automáticas. '
        'Use os critérios e confira evidências no PDF/OCR. A localização é página do PDF '
        'e face esquerda/direita, não página impressa. Nota 0–4 conforme PLANO.md. '
        'Oculte ou ignore estilo e referências na avaliação de conteúdo. '
        'O cegamento pode ser parcial, pois a redação pode revelar o modo. '
        'Verifique sustentação das citações em etapa separada usando responses.jsonl. '
        'Preserve o CSV preenchido com nome do avaliador e data. Não sobrescreva '
        'os resultados automáticos nem o gabarito congelado.\n')
    print(len(packet),'respostas no pacote de revisão')


if __name__=='__main__':main()
