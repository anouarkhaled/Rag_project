import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.3-70b-versatile"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> dict:
        """Retrieve the top-k passages for `query` and ask the LLM to answer
        citing which passage(s) it used, so the caller can show exactly which
        document/page backs the answer instead of a bare summary.
        """
        results = self.vectorstore.query(query, top_k=top_k)

        sources = []
        context_blocks = []
        for i, r in enumerate(results, start=1):
            meta = r["metadata"] or {}
            text = meta.get("text", "")
            if not text:
                continue
            source = meta.get("source", "unknown")
            page = meta.get("page")
            # PyPDFLoader pages are 0-indexed; show them as humans count pages.
            page_label = f", page {page + 1}" if isinstance(page, int) else ""
            context_blocks.append(f"[Passage {i} — {source}{page_label}]\n{text}")
            sources.append({
                "source": source,
                "page": (page + 1) if isinstance(page, int) else None,
                "excerpt": text[:240],
            })

        if not context_blocks:
            return {"summary": "No relevant documents found.", "sources": []}

        context = "\n\n".join(context_blocks)
        prompt = f"""Answer the question using only the numbered passages below. \
After each claim, cite the passage number(s) you used, e.g. "(Passage 2)". \
If the passages don't contain the answer, say so instead of guessing.

Question: {query}

{context}

Answer:"""
        response = self.llm.invoke([prompt])
        return {"summary": response.content, "sources": sources}

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "Ce quoi L'architecture informatique "
    result = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", result["summary"])
    print("Sources:", result["sources"])
