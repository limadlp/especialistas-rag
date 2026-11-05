"""Renderiza faces de evidência e cria índice local de auditoria, sem alterar PDF."""
import html
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from especialistas_rag import core


def main():
    import pypdfium2 as pdfium
    questions=core.read_jsonl(core.ROOT/'experiments/questions.jsonl')
    pages={p['id']:p for p in core.read_jsonl(core.DATA/'ocr/book.jsonl')}
    ids=sorted({i for q in questions for i in q['evidence_page_ids']})
    target=core.DATA/'ocr/audit'
    target.mkdir(exist_ok=True)
    doc=pdfium.PdfDocument(next((core.ROOT/'source_books').glob('skinner*.pdf')))
    parts=['<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Auditoria das fontes</title>',
           '<style>body{font:18px system-ui;max-width:1200px;margin:40px auto}article{border-top:1px solid #aaa;margin-top:40px} img{width:48%;vertical-align:top}pre{white-space:pre-wrap;display:inline-block;width:48%;font:17px Georgia;vertical-align:top}</style>',
           '<h1>Imagens e OCR das referências do questionário</h1><p>Comparar texto extraído e imagem. Nenhuma página impressa é inferida como confirmada.</p>']
    for identity in ids:
        p=pages[identity]
        im=doc[p['pdf_page']-1].render(scale=1.7).to_pil().convert('RGB')
        if p['side']=='left':im=im.crop((0,0,im.width//2,im.height))
        elif p['side']=='right':im=im.crop((im.width//2,0,im.width,im.height))
        im.save(target/f'{identity}.png')
        parts.append(f'<article id="{identity}"><h2>{identity}</h2><a href="{identity}.png"><img src="{identity}.png"></a><pre>{html.escape(p["text"])}</pre></article>')
    (target/'index.html').write_text('\n'.join(parts)+'\n</html>')
    print(len(ids),'faces de referência renderizadas em',target)


if __name__=='__main__':main()
