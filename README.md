# Basic RAG System

A production-oriented Retrieval-Augmented Generation (RAG)

## Milestone 1: Document Ingestion & Indexing

### Architecture

PDF → Document Loader → Chunker → Embedding Model
    → Qdrant Vector Store → Validation → Alias Activation

### Tech Stack

- Python 3.12
- LlamaIndex
- HuggingFace Embeddings: BAAI/bge-large-en-v1.5
- Qdrant
- Docker Compose
- Pytest & Ruff

### Requirements

- Python 3.12+
- Docker Engine + Docker Compose
- NVIDIA GPU recommended (CPU fallback available)

### Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -e '.[dev]'
```

Start Qdrant:

```bash
sudo systemctl start docker.service
sudo docker compose up -d
```

Place a PDF inside `data/raw/`.

### Run Ingestion

```bash
python -m rag_app.ingestion.cli --file data/raw/dspy.pdf
```

The pipeline builds a new versioned collection, validates its
point count, and activates it using a Qdrant alias.

### Tests

```bash
python -m pytest -q
python -m ruff check src tests
```

### Current Limitations

- Supports PDF ingestion only.
- Index validation currently checks point count.
- Old index versions require manual cleanup.
- Failed builds may leave incomplete collections.
- No retrieval, reranking, or response generation yet.

### Roadmap

1. Document Ingestion & Indexing
2. Retrieval & Reranking
3. End-to-End RAG with Ollama
4. FastAPI & Streamlit
5. Production Readiness & Evaluation