# RAG Project

Ce projet implémente un pipeline RAG (Retrieval-Augmented Generation) personnalisé avec une architecture modulaire basée sur FAISS et Sentence Transformers pour la recherche sémantique, et Groq LLM pour la génération de réponses.

## Architecture RAG

### 1. Pipeline d'Embeddings
- **Modèle** : Sentence Transformers (all-MiniLM-L6-v2)
- **Chunking** : Découpage des documents avec chevauchement (1000 tokens, 200 tokens d'overlap)
- **Stockage** : FAISS (IndexFlatL2) pour la recherche de similarité efficace
- **Persistance** : Sauvegarde des index FAISS et métadonnées sur disque

### 2. Retrieval
- Base vectorielle FAISS pour la recherche par similarité cosinus
- Requête par embeddings du texte utilisateur
- Récupération des top-k documents les plus pertinents (k=3 par défaut)
- Conservation des métadonnées et texte source

### 3. Generation
- **LLM** : Groq (llama-3.3-70b-versatile)
- **Prompt Template** : Format question-contexte-réponse
- Génération de résumé basé sur le contexte retrouvé

### 4. Interface
- API REST avec FastAPI
- Endpoints pour requêtes synchrones
- Support format JSON pour requêtes/réponses

## Structure du projet

```
Rag_project/
│
├── app.py
├── requirements.txt
├── data/                # Place tes documents ici
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── embedding.py
│   ├── search.py
│   ├── vectorstore.py
│   └── ...
└── main.py

```

## Installation

1. **Cloner le dépôt**  
2. **Créer un environnement virtuel**  
   ```powershell
   python -m venv env
   .\env\Scripts\activate
   ```
3. **Installer les dépendances**  
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Préparer les données

Place tes fichiers PDF, TXT, ou autres documents dans le dossier `data/`.

## Utilisation

### Indexation et recherche

- Pour indexer les documents et lancer une recherche :
  ```powershell
  python app.py
  ```

- Le script va :
  - Charger les documents depuis `data/`
  - Construire ou charger un index vectoriel (FAISS ou ChromaDB)
  - Effectuer une recherche RAG et afficher un résumé

### Exemple de code

```python
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

if __name__ == "__main__":
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    # store.build_from_documents(docs)
    store.load()
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
```

## Changer de backend vectoriel

- Le projet supporte FAISS et ChromaDB (voir `src/vectorstore.py`).
- Pour utiliser ChromaDB, assure-toi que `chromadb` est bien installé et adapte la classe correspondante.

## Dépendances principales

- langchain
- faiss-cpu
- sentence-transformers
- pymupdf, pypdf (pour la lecture PDF)
- python-dotenv (pour la gestion des clés API)

## Personnalisation

- Modifie `src/data_loader.py` pour adapter le chargement de tes documents.
- Modifie `src/embedding.py` pour changer de modèle d'embedding.
- Modifie `src/search.py` pour ajuster la logique de recherche ou de génération.

## API REST

Le projet expose une API REST avec FastAPI pour interroger le modèle :

### Démarrer le serveur

# Lancer le serveur
fastapi dev main.py 
### Endpoints

#### POST /call_llm/

Interroge le modèle avec une question.

**Request Body (JSON)**:
```json
{
    "query": "Quelle est votre question ?",
    "top_k": 3
}
```
- `query` (string, required): La question à poser
- `top_k` (integer, optional, default=3): Nombre de documents à utiliser pour le contexte

**Response (JSON)**:
```json
{
    "summary": "Réponse générée par le modèle..."
}
```

### Exemples d'utilisation

#### Avec curl (PowerShell)
```powershell
curl -X POST "http://localhost:8000/call_llm/" `
     -H "Content-Type: application/json" `
     -d "{\"query\": \"What is attention mechanism?\", \"top_k\": 3}"
```

#### Avec Python requests
```python
import requests

response = requests.post(
    "http://localhost:8000/call_llm/",
    json={
        "query": "What is attention mechanism?",
        "top_k": 3
    }
)
print(response.json()["summary"])
```

### Documentation API

- Documentation Swagger UI : http://localhost:8000/docs
- Documentation ReDoc : http://localhost:8000/redoc

## Configuration des clés API

### Configuration de Groq

1. Créez un fichier `.env` à la racine du projet
2. Ajoutez votre clé API Groq :
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

Le projet utilise Groq avec le modèle `llama-3.3-70b-versatile` pour la génération de réponses. Pour obtenir une clé API :
1. Créez un compte sur [Groq Cloud](https://console.groq.com)
2. Générez une nouvelle clé API dans les paramètres de votre compte
3. Copiez la clé dans votre fichier `.env`

## Notes

- Pour réindexer après ajout de documents, décommente la ligne `store.build_from_documents(docs)` dans `app.py`.