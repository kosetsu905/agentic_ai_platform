import chromadb
from sentence_transformers import SentenceTransformer
import subprocess


# =========================
# 1. 初始化
# =========================

# embedding model（和 ingest 一致）
model = SentenceTransformer("all-MiniLM-L6-v2")

# 向量数据库
client = chromadb.Client()
collection = client.get_collection(name="medical_docs")


# =========================
# 2. 检索
# =========================
def retrieve(query, k=5):
    query_embedding = model.encode([query])

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return documents, metadatas


# =========================
# 3. 调用 Ollama
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
# 4. RAG 主逻辑
# =========================
def rag_query(query):
    docs, metas = retrieve(query)

    context = "\n\n".join(docs)

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
# 5. CLI
# =========================
if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() == "exit":
            break

        answer = rag_query(query)
        print("\nAnswer:\n", answer)