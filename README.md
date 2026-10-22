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


## Milestone 2: Retrieval & Reranking

### Architecture

User Query → BGE Query Embedding → Qdrant Top 10
→ Cross-Encoder Reranking → Top 3 Chunks

### Run Retrieval

```bash
python -m rag_app.retrieval.cli \
  --query "What is DSPy?" \
  --top-k 10 \
  --top-n 3


## Milestone 3: End-to-End RAG Generation

### Architecture

User Query → Retrieval → Reranking → Context Builder → Prompt Template → Ollama → Answer + Sources

### LLM Configuration

- Provider: Ollama (local)
- Model: `llama3.2:1b`
- Temperature: `0.1`
- Context Window: `4096`
- GPU: NVIDIA RTX 4070 Ti SUPER

### Run the RAG Application

Start Ollama and Qdrant:

```bash
sudo systemctl start ollama.service
sudo systemctl start docker.service
sudo docker compose up -d
```

Run the RAG CLI:

```bash
python -m rag_app.generation.cli \
  --query "What is DSPy?"
```

The CLI returns a JSON object containing the generated answer and retrieved source metadata.

### Current Limitations

- Citation formatting is not consistently followed by the 1B model.
- Retrieved sources are not independently verified to support every claim.
- No explicit token-budget management.
- No automated answer-quality evaluation.
- Inference performance has not been benchmarked.
- The current implementation is a development baseline, not a production-ready service.