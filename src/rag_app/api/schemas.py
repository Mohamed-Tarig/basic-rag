from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=2000,
        description="Question to ask the RAG system",
    )


class SourceResponse(BaseModel):
    id: int
    file: str | None = None
    page: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]