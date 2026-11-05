"""Amostras de memória do sistema; não equivalem à memória exclusiva do modelo."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import sys
import httpx
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from especialistas_rag import core
from especialistas_rag.adapters.lmstudio import headers as lmstudio_headers

path=core.ROOT/'experiments/resources.jsonl'
log=core.ROOT/'build/experiment.log'
while True:
    if log.exists() and log.read_text().rstrip().endswith('PIPELINE CONCLUÍDO'): break
    record={'timestamp':datetime.now(timezone.utc).isoformat(),
            'system_memory':psutil.virtual_memory()._asdict(),
            'swap':psutil.swap_memory()._asdict()}
    try:
        inventory=httpx.get(core.LM_URL+'/api/v1/models',timeout=4,headers=lmstudio_headers(), trust_env=False).json()['models']
        record['loaded_models']=[m['key'] for m in inventory if m.get('loaded_instances')]
    except Exception as exc:record['inventory_error']=str(exc)
    processes=[]
    for p in psutil.process_iter(['pid','name','memory_info']):
        try:
            if any(t in p.info['name'].lower() for t in ['lm studio','lm-studio','llm_engine','llmster']):
                processes.append({'pid':p.pid,'name':p.info['name'],'rss_bytes':p.info['memory_info'].rss})
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    record['lmstudio_processes']=processes
    with path.open('a') as f:f.write(json.dumps(record)+'\n')
    time.sleep(30)
