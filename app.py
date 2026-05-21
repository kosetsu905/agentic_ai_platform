import asyncio
import logging

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from scripts.query import ask_question
from scripts.stt import transcribe

logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...)):
    """Transcribe uploaded audio to text using local faster-whisper."""
    audio_bytes = await audio.read()
    logger.info("Received audio: filename=%s, size=%d bytes", audio.filename, len(audio_bytes))

    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(None, transcribe, audio_bytes)

    return {"text": text}


@app.post("/rag")
async def rag(req: Request):
    body = await req.json()

    q = body.get("query", "")
    history = body.get("history", None)

    answer, docs = ask_question(q, history=history)

    return {
        "answer": answer,
        "docs": docs
    }
