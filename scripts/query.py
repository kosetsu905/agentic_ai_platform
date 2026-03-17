import chromadb
from sentence_transformers import SentenceTransformer
import subprocess


# =========================
# 1. Initialization
# =========================

# Load embedding model (must match ingest step)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to vector database
client = chromadb.Client()
collection = client.get_collection(name="medical_docs")


# =========================
# 2. Retrieve relevant documents
# =========================
def retrieve(query, k=5):
    # Convert query into embedding
    query_embedding = model.encode([query])

    # Perform similarity search
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return documents, metadatas


# =========================
# 3. Call local LLM via Ollama
# =========================
def ask_llm(prompt):
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout


# =========================
# 4. RAG pipeline
# =========================
def rag_query(query):
    docs, metas = retrieve(query)

    # Combine retrieved chunks into context
    context = "\n\n".join(docs)

    # Prompt engineering
    prompt = f"""
You are a medical assistant. Answer the question based ONLY on the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    answer = ask_llm(prompt)
    return answer


# =========================
# 5. CLI interface
# =========================
if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() == "exit":
            break

        answer = rag_query(query)
        print("\nAnswer:\n", answer)