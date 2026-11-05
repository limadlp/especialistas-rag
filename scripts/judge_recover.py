"""Retoma pares ausentes com o mesmo juiz/prompt e recupera campos explícitos.

Não inventa notas: aceita somente scores inteiros e booleanos inequívocos em A/B.
O motivo pode conter aspas internas que tornaram o JSON nativo malformado.
"""
import json as standard_json
from pathlib import Path
import re
from types import SimpleNamespace
import judge

last_recovered=False


def relaxed_loads(text):
    global last_recovered
    last_recovered=False
    try:
        return standard_json.loads(text)
    except standard_json.JSONDecodeError:
        matches=list(re.finditer(r'"([AB])"\s*:\s*\{',text))
        if len(matches)!=2 or {m[1] for m in matches}!={'A','B'}:
            raise ValueError('Formato inválido sem dois objetos A/B inequívocos')
        output={}
        for i,match in enumerate(matches):
            segment=text[match.end():matches[i+1].start() if i+1<len(matches) else len(text)]
            scores=re.findall(r'"score"\s*:\s*([0-4])\s*(?=,|\})',segment)
            flags=re.findall(r'"major_error"\s*:\s*(true|false)\s*(?=,|\})',segment)
            if len(scores)!=1 or len(flags)!=1:
                raise ValueError('Campos de nota/erro ausentes, ambíguos ou fora da escala')
            reason=re.search(r'"reason"\s*:\s*(.*)',segment,re.S)
            reason_text=reason[1].strip().rstrip(',').rstrip('}').strip().strip('"') if reason else '[motivo não recuperado]'
            output[match[1]]={'score':int(scores[0]),'major_error':flags[0]=='true',
                              'reason':'[Formato recuperado] '+reason_text}
        last_recovered=True
        return output


def dumps(value,*args,**kwargs):
    if isinstance(value,dict) and value.get('evaluation_type')=='automatic_exploratory':
        value={**value,'retry_of_missing_pair':True,'format_recovered':last_recovered,
               'parser':'judge_recover.py v1; critérios/prompt originais inalterados'}
    return standard_json.dumps(value,*args,**kwargs)


if __name__=='__main__':
    # Namespace-local proxy: does not modify Python's json module or core.read_jsonl.
    judge.json=SimpleNamespace(loads=relaxed_loads,dumps=dumps)
    judge.main()
