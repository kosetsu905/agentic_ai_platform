# Agentic AI Platform

This project is a local RAG application with:

- A FastAPI backend at `app.py`
- PDF ingestion and hybrid retrieval scripts in `scripts/`
- OpenSearch for keyword and vector search
- Ollama for local LLM generation
- A Next.js frontend in `frontend-ui/`
- Multi-turn conversation context with query rewriting

## Prerequisites

Install these tools before setting up the project:

- Python 3.10 or newer
- Node.js 20 or newer
- Docker Desktop
- Ollama
- FFmpeg (required for voice input transcription)

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key

`SERPER_API_KEY` is used by the web search wrapper in the backend. Do not commit real API keys.

## Multi-turn context (see "Multi-turn Conversation Context" section below)
CONTEXT_MAX_TURNS=5
CONTEXT_MAX_CHARS=2000
ENABLE_QUERY_REWRITE=true
```

## Voice input (optional, uses local Whisper)
```
WHISPER_MODEL=small          # tiny / base / small / medium / large, default: small
WHISPER_LANGUAGE=            # set to "zh" to force Chinese, empty = auto-detect
HF_ENDPOINT=https://hf-mirror.com  # Hugging Face mirror (useful in mainland China)
```

## Multi-turn Conversation Context

The system supports multi-turn conversations by passing chat history to the LLM
and using context-aware query rewriting to resolve pronoun references (e.g., "its", "they").

**How it works:**
- Each request carries the full conversation history from the frontend.
- The history is truncated (by turn count and character length) to stay within the LLM context window.
- When history is present, the current question is rewritten to be self-contained before retrieval.
- The truncated history is injected into the Prompt alongside retrieved documents.
- When no history is provided, the system falls back to single-turn mode (backward compatible).

**Configuration:**

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTEXT_MAX_TURNS` | `5` | Maximum conversation turns to retain. One turn = one user message + one assistant reply. |
| `CONTEXT_MAX_CHARS` | `2000` | Maximum characters of formatted chat history text injected into the Prompt. |
| `ENABLE_QUERY_REWRITE` | `true` | Whether to use LLM-based query rewriting for coreference resolution. Set to `false` to skip the extra LLM call (faster but may miss pronoun references). |

**Adjusting context size:**
- Edit the values in `.env` to increase or decrease context limits.
- Lower values reduce LLM latency and token usage; higher values provide richer context.
- The truncation applies two-stage filtering: rounds first, then character count (discarding the oldest messages first).
- Example: to retain only the last 3 turns and limit to 1000 chars:
  ```env
  CONTEXT_MAX_TURNS=3
  CONTEXT_MAX_CHARS=1000
  ```

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
