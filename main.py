from typing import Optional, List

from fastapi import FastAPI
from pydantic import BaseModel
from app import call_LLM

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class SourceRef(BaseModel):
    source: str
    page: Optional[int] = None
    excerpt: str


class QueryResponse(BaseModel):
    summary: str
    sources: List[SourceRef]


@app.post("/call_llm/", response_model=QueryResponse)
def call_llm_endpoint(req: QueryRequest):
    """POST endpoint to call the LLM. Expects JSON: {"query": "...", "top_k": 3}

    Returns JSON: {"summary": "...", "sources": [{"source": "...", "page": 3, "excerpt": "..."}]}
    """
    return call_LLM(req.query, top_k=req.top_k)
