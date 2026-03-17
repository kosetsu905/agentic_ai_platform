import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb

# =========================
# 1. Load PDF
# =========================
def load_pdf(path):
    docs = []
    pdf = fitz.open(path)

    for page_num, page in enumerate(pdf):
        text = page.get_text()
        if text.strip():
            docs.append({
                "content": text,
                "metadata": {
                    "source": path,
                    "page": page_num
                }
            })

    return docs


# =========================
# 2. Chunk
# =========================
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def chunk_documents(docs):
    chunked = []

    for doc in docs:
        chunks = chunk_text(doc["content"])

        for i, chunk in enumerate(chunks):
            chunked.append({
                "content": chunk,
                "metadata": {
                    **doc["metadata"],
                    "chunk_id": i
                }
            })

    return chunked


# =========================
# 3. Embedding + Store
# =========================
def main():
    # embedding model（本地）
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # vector DB（本地）
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="medical_docs")

    all_chunks = []

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            path = os.path.join(DATA_DIR, file)
            docs = load_pdf(path)
            chunks = chunk_documents(docs)
            all_chunks.extend(chunks)

    texts = [c["content"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    embeddings = model.encode(texts)

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=[str(i) for i in range(len(texts))]
    )

    print(f"Indexed {len(texts)} chunks.")


if __name__ == "__main__":
    main()