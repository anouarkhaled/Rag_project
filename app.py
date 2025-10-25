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

def call_LLM(query: str, top_k: int = 3) -> str:
     """Run the RAG search + summarization for `query` and return the summary.

     This function is safe to call from an API endpoint.
     """
     rag_search = _get_rag_search()
     return rag_search.search_and_summarize(query, top_k=top_k)


if __name__ == "__main__":
     # Simple CLI entrypoint for local testing
     rag = _get_rag_search()
     q = input("Enter your query: ")
     print(call_LLM(q, top_k=3))
