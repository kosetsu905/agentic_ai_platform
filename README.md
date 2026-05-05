# Agentic AI Platform

This project is a local RAG application with:

- A FastAPI backend at `app.py`
- PDF ingestion and hybrid retrieval scripts in `scripts/`
- OpenSearch for keyword and vector search
- Ollama for local LLM generation
- A Next.js frontend in `frontend-ui/`

## Prerequisites

Install these tools before setting up the project:

- Python 3.10 or newer
- Node.js 20 or newer
- Docker Desktop
- Ollama

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key
```

`SERPER_API_KEY` is used by the web search wrapper in the backend. Do not commit real API keys.

## Backend Setup

Create and activate a Python virtual environment:

```sh
python -m venv .venv
```

On Windows PowerShell:

```sh
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```sh
source .venv/bin/activate
```

Install Python dependencies:

```sh
pip install -r requirements.txt
```

## OpenSearch Setup

Pull and start OpenSearch:

```sh
docker pull opensearchproject/opensearch:latest
docker run -d --name opensearch -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -e "DISABLE_SECURITY_PLUGIN=true" opensearchproject/opensearch:latest
```

If the container already exists, start it with:

```sh
docker start opensearch
```

## Ollama Setup

Pull the local model used by the backend:

```sh
ollama pull llama3
```

Make sure the Ollama service is running before starting the backend.

## Ingest Documents

Place PDF files in the `data/` directory, then build the OpenSearch index:

```sh
python scripts/ingest.py
```

Run this again whenever the PDF corpus changes.

## Run the Backend

```sh
uvicorn app:app --reload
```

The backend listens on `http://127.0.0.1:8000`.

## Frontend Setup

Install Node dependencies from the lockfile:

```sh
cd frontend-ui
npm ci
```

Run the frontend:

```sh
npm run dev
```

The frontend listens on `http://localhost:3000` and calls the backend endpoint at `http://127.0.0.1:8000/rag`.

## Reproduction Notes

- Use `npm ci` instead of `npm install` to reproduce the exact frontend dependency tree from `package-lock.json`.
- Python dependencies are listed in `requirements.txt` with current major-version-compatible ranges. After validating a working environment, you can generate a fully pinned file with `pip freeze > requirements-lock.txt` if exact Python package versions are required.
- The embedding model is `sentence-transformers/all-MiniLM-L6-v2`.
- The reranker model is `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- The Ollama model is `llama3:latest`.
