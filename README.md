# Rebuilding Milo — RAG Chatbot

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/dependency--manager-poetry-2b9ed8.svg)](https://python-poetry.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

A local Retrieval-Augmented Generation (RAG) system that turns a text PDF (book) into a chatbot:
- Extracts and chunks text from a PDF
- Embeds chunks using OpenAI embeddings
- Stores vectors in Pinecone
- Serves a minimal chat UI & FastAPI RAG endpoint

This repo is prepared to run locally and via Docker so others can reproduce your demo easily.

---

## Quick features

- Ingest editable PDFs → `data/<book_slug>/chunks.jsonl`  
- Embedding & indexing → OpenAI + Pinecone  
- FastAPI `/rag` endpoint + single-file chat UI  
- Pluggable system prompt at `src/llm/prompt.py`  
- Poetry-managed, reproducible environment

---

## Repo layout

rebuilding-milo-chatbot/
├── samples/ # sample PDFs you can add
├── data/ # generated chunk files (ignored by git)
├── src/
│ ├── api/
│ ├── embeddings/
│ ├── ingestion/
│ ├── llm/
│ ├── pipeline/
│ └── vectorstore/
├── .env.example
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md


---

## Requirements (local dev)

- Python 3.13+
- Poetry
- OpenAI API key
- Pinecone API key

---

## Local developer quickstart
1. Clone & enter project:

```bash
git clone <your-repo-url>
cd rebuilding-milo-chatbot
```

2. Copy .env.example to .env and fill with your keys:

```bash
cp .env.example .env
# Edit .env: OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV (or PINECONE_ENVIRONMENT), PINECONE_INDEX
```

3. Install dependencies (Poetry):

```bash
poetry install
```

4. Ingest a PDF (example):

```bash
# change sample path to your PDF
poetry run python -m src.ingestion.ingest_pipeline "samples/your_book.pdf" --outdir data --max-tokens 750 --overlap-tokens 128
```

5. Create embeddings & index:

```bash
poetry run python -m src.pipeline.index_pipeline "data/<book_slug>/chunks.jsonl"
```

6. Run the server and open the UI:

```bash
poetry run uvicorn src.api.app:app --reload --port 8000
# open http://127.0.0.1:8000/ in browser
```

## Docker (recommended for reproducible demos)
**Files added**
* Dockerfile — multi-stage build using Poetry (see below)
* docker-compose.yml — convenience service to run UVicorn + mount data/ and .env
* .dockerignore — keep images small

**Build & run with Docker**

From repo root:
```bash

# build image
docker compose build

# run the app (reads .env in repo root; mounts data into container)
docker compose up -d

# tail logs
docker compose logs -f web
```

Open [http://127.0.0.1:8000/]

## Execute ingestion / indexer inside container

You can run one-off commands in the container to ingest or index:

```bash
# run ingestion (replace sample path)
docker compose run --rm web poetry run python -m src.ingestion.ingest_pipeline "samples/your_book.pdf" --outdir data --max-tokens 750 --overlap-tokens 128

# run indexer
docker compose run --rm web poetry run python -m src.pipeline.index_pipeline "data/<book_slug>/chunks.jsonl"
```

These commands write into the data/ folder on your host (mounted into container), so results persist outside the container.

## Environment variables

Create .env (or edit .env.example) with:

```bash
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pc-...
PINECONE_ENVIRONMENT=gcp-starter        # or appropriate env/region
PINECONE_INDEX=rebuilding-milo-index
EMBED_MODEL=text-embedding-3-small
BATCH_SIZE=64
LLM_MODEL=gpt-5                         # change according to your OpenAI access
PINECONE_NAMESPACE=default
EMBED_DIM=1536
```

**Security**: never commit .env or your keys. Use GitHub secrets for CI or private repo settings.

## Prompt customization

Edit the system prompt at:

```bash
src/llm/prompt.py
```

Only that file needs to change to tune system role / tone / citation style.

## Troubleshooting
* tiktoken not available — tokenizer falls back to whitespace; that's expected if you didn't install Rust toolchain. Not fatal.
* Pinecone errors — verify PINECONE_API_KEY and PINECONE_ENVIRONMENT.
* OpenAI errors — check OPENAI_API_KEY and that your account has access to the requested models.
* Docker permission issues on Windows — run Docker Desktop (with WSL2 backend recommended) and ensure file sharing is enabled for your project path.

## Contributing / License

MIT License. Feel free to fork and adapt.

Add issues or PRs for improvements or different vector DB backends.

## Acknowledgements

* Built for a demo of a RAG chatbot based on a book PDF.
* Uses OpenAI and Pinecone for embeddings & retrieval.