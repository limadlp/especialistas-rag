# Especialistas · RAG para bibliotecas pessoais

Autor: **Dangeles Lima**

[Read in English](README.en.md)

Um aplicativo local para fazer perguntas sobre coleções de PDF e EPUB. Cada especialista tem biblioteca, instruções e histórico próprios. As respostas exibem citações que abrem o trecho e a página do PDF ou a seção do EPUB.

O pipeline combina busca semântica e lexical, com embeddings multilíngues E5, Qdrant local, BM25 e Reciprocal Rank Fusion. Um LLM reordena os trechos, pode pedir uma segunda busca para preencher uma lacuna e gera a resposta com as evidências selecionadas. O frontend usa JavaScript nativo e conversa com o backend por HTTP JSON via FastAPI.

## Executar localmente

Requisitos: Python 3.12+, LM Studio com a API REST v1, espaço em disco para os pesos do modelo e Tesseract caso sejam necessários OCR e os idiomas `por` ou `eng`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

Baixe o modelo de embeddings uma vez. A indexação usa os pesos locais nas execuções seguintes. [Modelo multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)

```bash
python -m pip install huggingface_hub
hf download intfloat/multilingual-e5-small --local-dir data/models/multilingual-e5-small
```

1. Carregue um modelo de chat no LM Studio e inicie o servidor local na aba **Developer**. A primeira instância carregada é a seleção inicial. [API REST do LM Studio](https://lmstudio.ai/docs/developer/rest)
2. Execute `especialistas-rag platform-serve` e abra [a interface local](http://127.0.0.1:8001).
3. Selecione ou crie um especialista, importe um PDF ou EPUB e aguarde a indexação. Use **Atualizar modelos** ao trocar de modelo no LM Studio. A interface da API está no [Swagger](http://127.0.0.1:8001/docs).

## Configuração de modelos

| Variável | Uso |
| --- | --- |
| `TAC_LM_URL` | Endereço do LM Studio. O padrão é `http://127.0.0.1:1234`. |
| `LMSTUDIO_API_TOKEN` | Token quando a autenticação do LM Studio está ativada. |
| `GOOGLE_API_KEY` | Chave opcional do Gemini. `GEMINI_API_KEY` também é aceita. |
| `TAC_GOOGLE_MODEL` | ID de um modelo Gemini com suporte a chat compatível com OpenAI. Preencha junto com a chave. |
| `TAC_DATA` | Diretório para bibliotecas, índices e conversas. O padrão é `data/`. |
| `TAC_EMBEDDING_MODEL` | Caminho local dos pesos E5. O padrão é `<TAC_DATA>/models/multilingual-e5-small`. |

As credenciais são lidas pelo backend ao iniciar. Se você selecionar **Google Gemini** na interface, a API do Google recebe a pergunta, o contexto recente e os trechos utilizados na reordenação e na resposta. Uma chamada sem modelo explícito usa apenas o LM Studio. [Compatibilidade da API Gemini](https://ai.google.dev/gemini-api/docs/openai)

## Arquitetura

```mermaid
flowchart LR
    UI["Navegador · HTML, CSS e JavaScript"] -->|"HTTP JSON e upload"| API["FastAPI"]
    API -->|"Indexação PDF e EPUB"| DOC["Extração, OCR e divisão em trechos"]
    DOC --> IDX["Busca híbrida · Qdrant e BM25"]
    API --> RAG["Fluxo de RAG"]
    RAG --> IDX
    IDX --> E5["Embeddings E5 locais"]
    RAG --> LLM["Adaptadores de LLM"]
    LLM -->|"Padrão"| LM["LM Studio"]
    LLM -->|"Seleção explícita"| G["Google Gemini"]
    API --> SQLITE["Histórico · SQLite"]
```

| Código | Responsabilidade |
| --- | --- |
| [`web/api.py`](src/especialistas_rag/web/api.py) | Valida requisições, serve a interface e acompanha a indexação em segundo plano. |
| [`rag.py`](src/especialistas_rag/rag.py) | Seleciona evidências, refina a busca e gera respostas com citações. |
| [`adapters/`](src/especialistas_rag/adapters) | Integra PDF, EPUB, OCR, Qdrant, SQLite e provedores de LLM. |
| [`retrieval.py`](src/especialistas_rag/retrieval.py) | Carrega o encoder E5 e normaliza os termos da busca lexical. |
| [`settings.py`](src/especialistas_rag/settings.py) | Lê a configuração e as credenciais no backend. |
| [`evals/`](evals) | Gera o corpus descartável e mede recuperação, respostas e citações. |

```mermaid
sequenceDiagram
    actor Pessoa
    participant API as FastAPI
    participant RAG as Serviço RAG
    participant IDX as Biblioteca
    participant LLM as Modelo selecionado
    participant DB as SQLite

    Pessoa->>API: POST /api/chat
    API->>DB: Carrega histórico, quando existe
    API->>RAG: Busca evidências no especialista
    RAG->>IDX: Busca semântica e lexical
    opt Há trechos candidatos
        RAG->>LLM: Reordena os trechos
        opt O modelo identifica uma lacuna
            RAG->>IDX: Executa uma busca complementar
            RAG->>LLM: Reordena os candidatos combinados
        end
    end
    RAG->>LLM: Gera resposta com as evidências selecionadas
    API->>DB: Salva o turno e as citações
    API-->>Pessoa: Resposta, fontes e ID da conversa
```

## Avaliação

O benchmark **Cedro v2** usa **30 perguntas em português** sobre **24 trechos em oito documentos fictícios**. São seis perguntas de cada tipo: diretas, paráfrases, perguntas com trechos parecidos que podem confundir a busca, perguntas que exigem dois trechos e perguntas sem resposta no corpus. O gabarito identifica a seção exata do EPUB.

### Resultado da busca híbrida

Execução local com multilingual-e5-small, Qdrant e BM25. As métricas abaixo são calculadas nas **24 perguntas respondíveis**, antes da reordenação pelo LLM.

| Posição | Pelo menos um trecho correto | Todos os trechos exigidos |
| --- | ---: | ---: |
| Top 1 | 20/24 (83,3%) | 15/24 (62,5%) |
| Top 3 | 23/24 (95,8%) | 22/24 (91,7%) |
| Top 5 | 24/24 (100%) | 23/24 (95,8%) |

O **MRR@5 foi 0,897**. No top 5, a busca recuperou as duas evidências em **5/6** perguntas que exigiam dois trechos. Na pergunta restante, sobre o limite de chuva no piloto e na norma vigente, só apareceu a norma vigente. Os resultados medem regressões nesse corpus controlado.

```bash
python -m pytest -q
python -m evals.run > /tmp/cedro-v2.json
```

O avaliador importa os EPUBs em um diretório temporário e devolve as posições por pergunta, métricas por tipo e o SHA-256 dos pesos usados. Com um modelo carregado no LM Studio, `python -m evals.run --generate` também compara respostas com e sem RAG, checa citações e mede abstenções nas seis perguntas sem evidência. A checagem de respostas usa variantes de texto aceitas e pode perder paráfrases bem formuladas.

### Avaliação de respostas com Gemma 4 E4B

Uma execução local com `google/gemma-4-e4b` no LM Studio comparou o mesmo modelo nas 30 perguntas, com e sem acesso aos trechos. Os resultados abaixo foram conferidos nas respostas e nas fontes citadas.

| Métrica | Sem RAG | Com RAG |
| --- | ---: | ---: |
| Respostas corretas nas perguntas respondíveis | 0/24 | 24/24 |
| Respostas corretas que citaram todos os trechos exigidos | — | 24/24 |
| Abstenções corretas nas perguntas sem evidência | 6/6 | 5/6 |

Sem acesso ao corpus fictício, o modelo se absteve nas 30 perguntas. Com RAG, no único erro de abstenção, citou o dia em que a contestação é analisada ao ser perguntado sobre o prazo para contestar. Esse resultado descreve uma execução no Cedro v2.

## Dados locais

O Git ignora `.env`, livros, uploads, conversas, pesos, índices e resultados. A árvore começa apenas com o código e as avaliações que geram suas próprias entradas. Os diretórios `data/`, `source_books/`, `experiments/`, `build/` e `output/` contêm material local, incluindo o corpus e os artefatos do experimento original. O app preserva o formato das bibliotecas e conversas existentes.

Rascunhos e anotações locais ficam em `local/docs/`, que também é ignorado pelo Git.

O servidor é voltado ao uso em `127.0.0.1`. Ele mantém um índice Qdrant embutido por especialista e executa a indexação em segundo plano no processo da API.
