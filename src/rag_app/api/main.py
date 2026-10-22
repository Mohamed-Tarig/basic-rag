import logging

from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool

from rag_app.api.schemas import QueryRequest, QueryResponse
from rag_app.api.upload import router as upload_router
from rag_app.generation.pipeline import generate_answer

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Basic RAG API",
    version="0.1.0",
    description="REST API for the Basic RAG system",
)

app.include_router(upload_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    if not request.query.strip():
        raise HTTPException(
            status_code=422,
            detail="Query cannot contain only whitespace",
        )

    try:
        result = await run_in_threadpool(
            generate_answer,
            query=request.query,
        )
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(
            status_code=503,
            detail="RAG service temporarily unavailable",
        ) from exc

    return QueryResponse.model_validate(result)