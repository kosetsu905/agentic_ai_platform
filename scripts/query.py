import os
import chromadb
from sentence_transformers import SentenceTransformer
import subprocess


model = SentenceTransformer("all-MiniLM-L6-v2")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

print("DB PATH:", DB_DIR)

client = chromadb.PersistentClient(path=DB_DIR)

try:
    collection = client.get_collection(name="medical_docs")
except Exception:
    raise RuntimeError("Collection not found. Run ingest.py first.")


def retrieve(query, k=5):
    query_embedding = model.encode([query])

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )

    return results["documents"][0], results["metadatas"][0]


def ask_llm(prompt):
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        capture_output=True,
        encoding="utf-8"
    )
    return result.stdout


def rag_query(query):
    docs, _ = retrieve(query)
    context = "\n\n".join(docs)

    prompt = f"""
You are a medical assistant. Answer ONLY using the context.

Context:
{context}

Question:
{query}

Answer:
"""

    return ask_llm(prompt)


if __name__ == "__main__":
    while True:
        q = input(">> ")
        if q.lower() == "exit":
            break

        print(rag_query(q))