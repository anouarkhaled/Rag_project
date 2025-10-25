from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from app import call_LLM

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


@app.post("/call_llm/")
def call_llm_endpoint(req: QueryRequest):
    """POST endpoint to call the LLM. Expects JSON: {"query": "...", "top_k": 3}

    Returns JSON: {"summary": "..."}
    """
    summary = call_LLM(req.query, top_k=req.top_k)
    return {"summary": summary}
