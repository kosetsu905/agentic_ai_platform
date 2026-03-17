import os
import fitz
from sentence_transformers import SentenceTransformer
import chromadb


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


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
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


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    DB_DIR = os.path.join(BASE_DIR, "chroma_db")

    print("DB PATH:", DB_DIR)

    client = chromadb.PersistentClient(path=DB_DIR)

    collection = client.get_or_create_collection(name="medical_docs")

    all_chunks = []

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

    print("Indexed:", len(texts))


if __name__ == "__main__":
    main()