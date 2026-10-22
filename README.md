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


## Milestone 4: FastAPI and Streamlit Application

### Architecture

The application provides a web interface and REST API on top of the existing RAG pipeline.

```text
Streamlit UI (:8501)
       |
       v
FastAPI (:8000)
       |
       +-- GET /health
       |
       +-- POST /query
       |      |
       |      +-- Retrieval -> Reranking -> Ollama -> Answer
       |
       +-- POST /documents/upload
              |
              +-- PDF Validation
              +-- Document Loading & Chunking
              +-- Embedding & Versioned Qdrant Indexing
              +-- Index Validation
              +-- Atomic Alias Activation
```

### Running the Application

Start the required services:

```bash
sudo systemctl start docker.service
sudo docker compose up -d
sudo systemctl start ollama.service
```

Start FastAPI:

```bash
python -m uvicorn rag_app.api.main:app \
  --host 127.0.0.1 --port 8000
```

Start Streamlit in another terminal:

```bash
python -m streamlit run src/rag_app/ui/app.py \
  --server.address 127.0.0.1 --server.port 8501
```

Web interface: `http://127.0.0.1:8501`

Interactive API documentation: `http://127.0.0.1:8000/docs`

### PDF Upload

The application supports uploading one PDF at a time.

- Maximum file size: 20 MB.
- Validates the `.pdf` extension and PDF header.
- Builds a new versioned Qdrant collection.
- Validates the stored vector count before activating the new collection.
- Preserves previous collections for possible rollback.
- **Uploading a new PDF replaces the active knowledge base; it does not append documents.**

For isolated upload testing, start FastAPI with:

```bash
RAG_UPLOAD_COLLECTION_PREFIX=basic_rag_upload_test \
python -m uvicorn rag_app.api.main:app \
  --host 127.0.0.1 --port 8000
```

This activates `basic_rag_upload_test_active` without changing `basic_rag_active`. The `/query` endpoint continues using the original active index.

### Verification

- 71 automated tests passed.
- FastAPI query endpoint tested against the running RAG pipeline.
- Real PDF upload tested through both HTTP and Streamlit.
- Document indexing and retrieval verified in Qdrant.
- Original active collection preserved during isolated upload testing.

### Current Limitations

- Single-PDF knowledge base; multi-document support is deferred.
- Citation compliance and factual grounding are not independently validated.
- Upload processing is synchronous from the user's perspective and may take time.
- No authentication, authorization, or concurrent upload coordination.
- No automated end-to-end regression tests with live infrastructure.
- The application is a development baseline and is not ready for public deployment.