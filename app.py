from src.data_loader import load_all_documents
from src.search import RAGSearch
# Lazy-initialized RAGSearch instance so importing this module doesn't
# block or perform heavy work until the first API/CLI call.
_rag_search = None

def _get_rag_search():
     global _rag_search
     if _rag_search is None:
          # RAGSearch will build or load the vectorstore as needed
          _rag_search = RAGSearch()
     return _rag_search

def call_LLM(query: str, top_k: int = 3) -> dict:
     """Run the RAG search + summarization for `query` and return a dict with
     the generated answer and the list of source passages (file + page) it
     was grounded in.

     This function is safe to call from an API endpoint.
     """
     rag_search = _get_rag_search()
     return rag_search.search_and_summarize(query, top_k=top_k)


if __name__ == "__main__":
     # Simple CLI entrypoint for local testing
     rag = _get_rag_search()
     q = input("Enter your query: ")
     result = call_LLM(q, top_k=3)
     print("\nAnswer:", result["summary"])
     if result["sources"]:
          print("\nSources:")
          for s in result["sources"]:
               page = f", page {s['page']}" if s["page"] is not None else ""
               print(f"  - {s['source']}{page}")
