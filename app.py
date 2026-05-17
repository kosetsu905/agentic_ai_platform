from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from scripts.query import ask_question

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
