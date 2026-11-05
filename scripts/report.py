import csv
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from especialistas_rag import core


def bootstrap(deltas, n=10000):
    rng = np.random.default_rng(20261030)
    values = np.asarray(deltas, dtype=float)
    if not len(values): return None
    samples = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.percentile(samples, [2.5,97.5])]


def main():
    out=core.ROOT/'output'
    out.mkdir(exist_ok=True)
    responses=core.read_jsonl(core.ROOT/'experiments/responses.jsonl')
    judgments=core.read_jsonl(core.ROOT/'experiments/judgments.jsonl') if (core.ROOT/'experiments/judgments.jsonl').exists() else []
    if len(responses)!=648 or len(judgments)!=324:
        raise SystemExit(f'Avaliação incompleta: {len(responses)}/648 respostas, {len(judgments)}/324 pares. Retome scripts/pipeline.py antes de publicar o relatório.')
    scores={(r['question_id'],r['model_alias'],r['repetition'],mode):s
            for r in judgments for mode,s in r['scores'].items()}
    questions={q['id']:q for q in core.read_jsonl(core.ROOT/'experiments/questions.jsonl')}
    summary={'evaluation_type':'automatic_exploratory','expert_review':'not_performed','codex_review':{'type':'qualitative_unblinded_posthoc','responses':48,'file':'output/REVISAO_CODEX.md'},
             'expected_final_responses':648,'observed_responses':len(responses),'conditions':[], 'gains':[]}
    flat=[]
    for row in responses:
        s=scores.get((row['question_id'],row['model_alias'],row['repetition'],row['mode']),{})
        flat.append({'question_id':row['question_id'],'category':row['category'],'model':row['model_alias'],
                     'mode':row['mode'],'repetition':row['repetition'],'status':row['status'],
                     'score_0_4':s.get('score'), 'major_error':s.get('major_error'),
                     'seconds':row.get('generation_seconds'),'output_tokens':row.get('stats',{}).get('total_output_tokens'),
                     'reasoning_tokens':row.get('stats',{}).get('reasoning_output_tokens'),
                     'possibly_truncated':row.get('possibly_truncated'), 'answer':row.get('answer','')})
    with (out/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    for alias in core.MODELS:
        for mode in ['baseline','rag']:
            rs=[r for r in flat if r['model']==alias and r['mode']==mode]
            nums=[r['score_0_4'] for r in rs if r['score_0_4'] is not None]
            secs=[r['seconds'] for r in rs if r['seconds'] is not None]
            summary['conditions'].append({'model':alias,'mode':mode,'runs':len(rs),'scored':len(nums),
                'score_100':float(np.mean(nums)*25) if nums else None,
                'major_errors':sum(r['major_error'] is True for r in rs),
                'median_seconds':float(np.median(secs)) if secs else None,
                'p95_seconds':float(np.percentile(secs,95)) if secs else None,
                'failures':sum(r['status']!='ok' for r in rs),
                'possibly_truncated':sum(bool(r['possibly_truncated']) for r in rs)})
        byquestion={}
        for j in judgments:
            if j['model_alias']==alias:
                byquestion.setdefault(j['question_id'],[]).append((j['scores']['rag']['score']-j['scores']['baseline']['score'])*25)
        differences=[float(np.mean(v)) for v in byquestion.values()]
        summary['gains'].append({'model':alias,'questions':len(differences),
            'gain_points':float(np.mean(differences)) if differences else None,
            'bootstrap95':bootstrap(differences),
            'wins':sum(v>0 for v in differences),'ties':sum(v==0 for v in differences),'losses':sum(v<0 for v in differences)})
    frozen=core.read_jsonl(core.ROOT/'experiments/frozen_sources.jsonl')
    retrieval=[]
    for r in frozen:
        q=questions[r['question_id']]
        if q['split']!='test' or q['id'] in {'test-32','test-33'}:continue
        pages={s['page_id'] for s in r['sources']}
        reference=set(q['evidence_page_ids'])
        retrieval.append({'question_id':q['id'],'category':q['category'],
                          'reference_page_hit_at_5':bool(reference & pages),
                          'reference_page_coverage_at_5':len(reference & pages)/len(reference)})
    summary['retrieval']={'reference_page_hit_at_5':float(np.mean([r['reference_page_hit_at_5'] for r in retrieval])),
                           'questions':len(retrieval),'excluded':['test-32','test-33'],
                           'note':'Proxy de páginas de referência, não mede cobertura completa de evidências nem relevância de todas as fontes.'}
    citations=[]
    for r in responses:
        if r['mode']!='rag' or r['status']!='ok':continue
        cited=set(re.findall(r'pdf\d{3}-(?:left|right|single)-c\d+',r['answer']))
        allowed={s['id'] for s in r['sources']}
        citations.append({'valid':len(cited & allowed),'invalid':len(cited-allowed),'citations':len(cited)})
    summary['citation_ids']={'valid':sum(r['valid'] for r in citations),'invalid':sum(r['invalid'] for r in citations),
                               'note':'Existência do identificador não comprova que a fonte sustente a afirmação.'}
    core.write_jsonl(out/'retrieval_metrics.jsonl',retrieval)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    categories=[]
    for alias in core.MODELS:
        for category in ['conceitos','aplicacao','livro','limites']:
            item={'model':alias,'category':category}
            for mode in ['baseline','rag']:
                values=[r['score_0_4']*25 for r in flat if r['model']==alias and r['category']==category and r['mode']==mode and r['score_0_4'] is not None]
                item[mode]=float(np.mean(values)) if values else None
            categories.append(item)
    summary['categories']=categories
    resource_path=core.ROOT/'experiments/resources.jsonl'
    resource_content=resource_path.read_text() if resource_path.exists() else ''
    if resource_content and not resource_content.endswith('\n'):
        resource_content=resource_content.rsplit('\n',1)[0] if '\n' in resource_content else ''
    resource_rows=[json.loads(line) for line in resource_content.splitlines() if line.strip()]
    if responses:
        end=max(datetime.fromisoformat(r['timestamp']).timestamp()+r.get('generation_seconds',0) for r in responses)
        resource_rows=[r for r in resource_rows if datetime.fromisoformat(r['timestamp']).timestamp()<=end]
    resource_summary=[]
    for alias,key in core.MODELS.items():
        samples=[r for r in resource_rows if r.get('loaded_models')==[key]]
        resource_summary.append({'model':alias,'samples':len(samples),
            'min_system_available_gib':min((r['system_memory']['available']/1024**3 for r in samples),default=None),
            'median_system_available_gib':float(np.median([r['system_memory']['available']/1024**3 for r in samples])) if samples else None,
            'max_system_swap_used_gib':max((r['swap']['used']/1024**3 for r in samples),default=None)})
    summary['resources']={'by_model':resource_summary,'note':'Memória disponível e swap do sistema, não memória exclusiva do modelo. Amostras de 30s iniciadas durante a primeira rodada do 12B.'}
    summary['reasoning_tokens']=sum(r.get('stats',{}).get('reasoning_output_tokens',0) for r in responses)
    error_log=core.ROOT/'build/judge-errors.jsonl'
    retry_keys={tuple(r['key']) for r in core.read_jsonl(error_log)} if error_log.exists() else set()
    summary['judgment_retries']=sum(j.get('retry_of_missing_pair',False) or (j['question_id'],j['model_alias'],j['repetition']) in retry_keys for j in judgments)
    summary['judgment_format_recovered']=sum(j.get('format_recovered',False) for j in judgments)
    summary['environment']=json.loads((core.ROOT/'experiments/environment.json').read_text())
    profile_path=core.ROOT/'output/retrieval_profile.json'
    if profile_path.exists():
        profile=json.loads(profile_path.read_text())
        summary['auxiliary_retrieval_profile']={k:v for k,v in profile.items() if k!='samples'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    make_report(summary, responses, judgments)
    make_charts(summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def make_report(s, responses, judgments):
    lines=['# Experimento local: Gemma com e sem RAG', '',
           '**Notas automáticas exploratórias. Revisão qualitativa do Codex em 48 respostas, registrada separadamente.**','',
           f"Execuções registradas: {s['observed_responses']} de 648 planejadas. Julgamentos pareados: {len(judgments)} de 324.", '',
           '## Método','',
           'Fonte única: Ciência e comportamento humano, de Skinner. OCR local em português de 255 páginas digitalizadas, separadas em 510 faces. Índice: 940 trechos de até 320 tokens, sobreposição de 50 tokens, E5 Small e busca híbrida vetorial/BM25 com RRF. Cinco trechos congelados por pergunta, iguais para os três modelos.', '',
           'Três modelos instalados: Gemma 4 E4B, 12B e 26B A4B QAT, todos MLX 4bit. Mesmo prompt, limite de 320 tokens, temperatura 0,1, contexto 8192, reasoning off e conversa independente. Até duas requisições simultâneas ao mesmo modelo; latência inclui espera sob essa carga. Smoke test sequencial separado. O 26B é MoE/QAT: diferenças não podem ser atribuídas somente ao número de parâmetros.', '',
           '36 perguntas finais: 10 conceituais, 10 aplicações hipotéticas, 8 específicas do livro e 8 limites/premissas falsas. 12 perguntas dev separadas. Três repetições por cenário. Sem gabarito ou questões no índice.', '',
           'Juiz automático: o Gemma 4 26B A4B QAT já instalado, sem thinking, recebe gabarito/evidências e pares A/B em ordem aleatória, sem identidade do modelo e sem indicação da condição. A nota é 0–4, normalizada para 0–100. Há viés possível do juiz, inclusive por avaliar respostas de seu próprio modelo. O cegamento é parcial: a redação pode revelar o uso de fontes. Não se trata de precisão clínica validada.', '',
           f"Julgamentos retomados após falha de processamento: {s['judgment_retries']}; destes, {s['judgment_format_recovered']} tiveram campos numéricos explícitos recuperados de JSON malformado. O prompt e os critérios do juiz foram preservados. As respostas originais dos modelos não foram alteradas.", '',
           '## Resultados automáticos','',
           '| Modelo | Modo | Execuções | Avaliadas | Nota /100 | Latência mediana (s) | Falhas | Possível truncamento |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in s['conditions']:
        score='pendente' if r['score_100'] is None else f"{r['score_100']:.1f}"
        sec='—' if r['median_seconds'] is None else f"{r['median_seconds']:.1f}"
        lines.append(f"| {r['model']} | {r['mode']} | {r['runs']} | {r['scored']} | {score} | {sec} | {r['failures']} | {r['possibly_truncated']} |")
    lines+=['','## Ganho pareado do RAG','', '| Modelo | Ganho em pontos | IC bootstrap 95% | Vitórias / empates / derrotas por pergunta |','|---|---:|---|---|']
    for r in s['gains']:
        gain='pendente' if r['gain_points'] is None else f"{r['gain_points']:+.1f}"
        ci='pendente' if r['bootstrap95'] is None else f"[{r['bootstrap95'][0]:+.1f}; {r['bootstrap95'][1]:+.1f}]"
        lines.append(f"| {r['model']} | {gain} | {ci} | {r['wins']} / {r['ties']} / {r['losses']} |")
    lines+=['','Intervalos reamostram questões, mantendo juntas as repetições. São condicionais ao questionário e ao juiz automático; não incluem toda a incerteza do julgamento. Comparações secundárias entre modelos não têm correção formal de multiplicidade.', '',
            '## Resultados por categoria','',
            '| Modelo | Categoria | Sem RAG /100 | Com RAG /100 |','|---|---|---:|---:|']
    for row in s['categories']:
        a='pendente' if row['baseline'] is None else f"{row['baseline']:.1f}"
        b='pendente' if row['rag'] is None else f"{row['rag']:.1f}"
        lines.append(f"| {row['model']} | {row['category']} | {a} | {b} |")
    lines+=['','## Ambiente e recursos','',
            'LM Studio 0.4.25+1; CLI commit 69d945a; runtime selecionado MLX NAX 1.13.1. Apple M5 com 24 GiB de memória unificada. O servidor informa quatro slots, mas o experimento mantém no máximo duas requisições em andamento.', '',
            f"Tokens de reasoning registrados nas respostas: {s['reasoning_tokens']}.", '',
            '| Modelo | Amostras | Menor memória disponível no sistema (GiB) | Maior swap usado no sistema (GiB) |','|---|---:|---:|---:|']
    for row in s['resources']['by_model']:
        available='—' if row['min_system_available_gib'] is None else f"{row['min_system_available_gib']:.1f}"
        swap='—' if row['max_system_swap_used_gib'] is None else f"{row['max_system_swap_used_gib']:.1f}"
        lines.append(f"| {row['model']} | {row['samples']} | {available} | {swap} |")
    lines+=['','Essas amostras incluem o restante do sistema e caches. Não representam a memória exclusiva do Gemma, nem permitem atribuir todo o swap ao experimento. Já havia swap na primeira amostra. O monitor começou durante a primeira rodada do 12B e continuou nas seguintes.', '',
            '## Recuperação e citações','', f"Acerto de pelo menos uma página de referência entre os cinco trechos: {s['retrieval']['reference_page_hit_at_5']:.1%}, em {s['retrieval']['questions']} perguntas. Excluídas test-32 (dose de medicamento) e test-33 (eficácia de chatbot), que exigem reconhecer falta de base para a afirmação solicitada. É uma aproximação: outras páginas podem ser relevantes e as páginas de referência não são um conjunto exaustivo.", '',
            f"Identificadores de fontes citados existentes no contexto: {s['citation_ids']['valid']}; inexistentes: {s['citation_ids']['invalid']}. Isso não verifica sustentação semântica de cada afirmação.", '',
            '## Medição auxiliar da busca','']
    if s.get('auxiliary_retrieval_profile'):
        p=s['auxiliary_retrieval_profile']
        lines += [f"Mediana: {p['median_seconds']*1000:.1f} ms; p95: {p['p95_seconds']*1000:.1f} ms em {p['questions']} consultas. Inicialização do retriever: {p['initialization_seconds']:.1f} s. A medição ocorreu após a geração principal, com o juiz em execução, e não representa latência ponta a ponta da API. Fontes reproduzidas exatamente: {p['all_sources_reproduced']}.",'',
                  'Diagnóstico da test-01: a página de definição empírica aparece em 16º/21º nos candidatos vetoriais e em 43º no ranking lexical, fora da seleção final de cinco trechos. O diagnóstico foi feito após a coleta, sem alterar o índice ou os resultados. Está em retrieval_profile.json.','']
    lines += ['## Limitações e interpretação','',
            '- Referências e critérios têm curadoria técnica; não houve revisão independente por especialista em análise do comportamento.',
            '- As 44 faces usadas como referências dev/test foram conferidas visualmente por agente; não houve revisão editorial integral do OCR nem revisão humana especializada independente.',
            '- A fidelidade a uma fonte histórica não demonstra atualidade ou adequação clínica de todas as suas afirmações e exemplos.',
            '- Há questões sobre temas semelhantes; os intervalos por pergunta podem subestimar dependências temáticas e são exploratórios.',
            '- Skinner pode estar representado no treinamento prévio dos modelos; o baseline não é um modelo sem conhecimento do livro.',
            '- Perguntas de premissas falsas e algumas vinhetas podem favorecer cautela já solicitada no prompt comum.',
            '- O pequeno conjunto e eventual efeito teto limitam generalização; perguntas específicas do livro favorecem acesso à fonte.',
            '- O questionário cobre um recorte do livro, sobretudo conceitos fundamentais; não é avaliação exaustiva de todos os capítulos.',
            '- Pontuação de juiz LLM pode conter erros e viés; conferir exemplos e revisar o pacote humano antes de tratar o ganho como estabelecido.',
            '- A verificação automática de IDs não mede a sustentação semântica de cada citação. Fidelidade detalhada e classificação independente de abstenções ficam para a revisão humana.',
            '- Latência depende da máquina e do aquecimento. Não representa benchmark universal; memória do processo Python não mede todo o uso de Metal/LM Studio.',
            '- A máquina continuou em uso durante a coleta, sem isolamento de laboratório. Tempos e memória são medidas descritivas desse ambiente.',
            '- O resultado não demonstra eficácia terapêutica nem capacidade de conduzir atendimento.', '',
            '## Revisão técnica por IA','', 'O Codex revisou qualitativamente 48 respostas, oito questões nas seis condições, primeira repetição. Seleção intencional posterior aos resultados e sem cegamento. Não é revisão humana, independente ou validação clínica, nem estima taxa geral de erros. Consulte output/REVISAO_CODEX.md e experiments/codex_review.jsonl. As notas automáticas originais foram preservadas.', '', '## Auditoria técnica amostral','',
            'Na questão test-01, repetição 1 do 12B com RAG, a resposta agrupa dinheiro como exemplo de reforçador primário. A página impressa 88 o apresenta como generalizado. O juiz atribuiu 3/4 e não apontou esse detalhe na justificativa. O caso foi registrado separadamente para revisão; as notas automáticas originais foram preservadas. Essa amostra não estima a taxa de erros do juiz.', '',
            f"[Auditoria técnica detalhada]({core.ROOT/'output/AUDITORIA_TECNICA.md'}). Revisão por agente, não humana nem independente do autor do experimento.", '',
            '## Reproduzir e revisar','',
            'Consulte README.md, experiments/config.json, experiments/manifest.json, experiments/questions.jsonl, experiments/frozen_sources.jsonl, experiments/responses.jsonl e experiments/judgments.jsonl. Os hashes congelam fonte processada, índice, perguntas e código de geração. O arquivo de resultados CSV permite revisão independente.', '',
            'Gráficos: quality.png e latency.png, gerados a partir de summary.json.','']
    (core.ROOT/'output/RELATORIO.md').write_text('\n'.join(lines))


def make_charts(s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric, title, ylabel, filename in [
        ('score_100','Nota automática exploratória — juiz Gemma','Nota (0–100)','quality.png'),
        ('median_seconds','Latência no Apple M5 / 24 GB — concorrência 2','Segundos, mediana','latency.png')]:
        fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained')
        positions=np.arange(3)
        for offset,mode,color in [(-.18,'baseline','#697b90'),(.18,'rag','#147d78')]:
            values=[next(r[metric] for r in s['conditions'] if r['model']==alias and r['mode']==mode) for alias in core.MODELS]
            values=[np.nan if v is None else v for v in values]
            bars=ax.bar(positions+offset,values,.34,label='Sem RAG' if mode=='baseline' else 'Com RAG',color=color)
            ax.bar_label(bars,fmt='%.1f',padding=3)
        ax.set_xticks(positions,['E4B','12B','26B A4B QAT'])
        ax.set_title(title,pad=18);ax.set_ylabel(ylabel);ax.legend(frameon=False)
        if metric=='score_100':ax.set_ylim(0,110)
        ax.spines[['top','right']].set_visible(False)
        fig.savefig(core.ROOT/'output'/filename,dpi=160)
        plt.close(fig)


if __name__=='__main__':main()
