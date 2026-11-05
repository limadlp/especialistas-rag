"""Executa geração, julgamento exploratório e relatório, com arquivos retomáveis."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
for script,args in [('run_experiment.py',['--repetitions','3']),('judge.py',[]),('judge_recover.py',[]),('report.py',[])]:
    print('ETAPA',script,flush=True)
    subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,check=True)
print('PIPELINE CONCLUÍDO',flush=True)
