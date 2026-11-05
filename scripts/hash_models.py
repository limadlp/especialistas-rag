"""Manifesto de pesos existentes; executar depois da medição de desempenho."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=Path.home()/'.lmstudio/models/lmstudio-community'
folders=['gemma-4-E4B-it-MLX-4bit','gemma-4-12B-it-MLX-4bit','gemma-4-26B-A4B-it-QAT-MLX-4bit']
records=[]
for folder in folders:
    for path in sorted((base/folder).iterdir()):
        if path.suffix not in {'.json','.jinja','.safetensors'}:continue
        with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        records.append({'model_folder':folder,'file':path.name,'bytes':path.stat().st_size,'sha256':sha})
        print(folder,path.name,flush=True)
(ROOT/'experiments/model_files.json').write_text(json.dumps(records,indent=2))
embedding=ROOT/'data/models/multilingual-e5-small/model.safetensors'
with embedding.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
(ROOT/'experiments/embedding_weights.json').write_text(json.dumps({'model':'intfloat/multilingual-e5-small','bytes':embedding.stat().st_size,'sha256':sha},indent=2))
