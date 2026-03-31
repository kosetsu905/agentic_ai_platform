import os
import fitz
import uuid
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from opensearchpy import OpenSearch, helpers


def load_pdf(path):
    docs = []
    pdf = fitz.open(path)
    filename = os.path.basename(path)

    for page_num, page in enumerate(pdf):
        text = page.get_text()
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": page_num,
                        "doc_type": "medical"
                    }
                )
            )
    return docs


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    documents = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            path = os.path.join(DATA_DIR, file)
            documents.extend(load_pdf(path))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    client = OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        use_ssl=False
    )

    index_name = "medical_docs"

    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)

    mapping = {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 3,
                "number_of_replicas": 1
            }
        },
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "metadata": {
                    "properties": {
                        "source": {"type": "keyword"},
                        "page": {"type": "integer"},
                        "doc_type": {"type": "keyword"}
                    }
                },
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 384,
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss",
                        "space_type": "cosinesimil"
                    }
                }
            }
        }
    }

    client.indices.create(index=index_name, body=mapping)

    actions = []
    texts = [d.page_content for d in docs]
    vectors = embeddings.embed_documents(texts)

    for d, vector in zip(docs, vectors):
        actions.append({
            "_index": index_name,
            "_id": str(uuid.uuid4()),
            "_source": {
                "content": d.page_content,
                "metadata": d.metadata,
                "embedding": vector
            }
        })

    helpers.bulk(client, actions)
    print(f"Indexed {len(docs)} chunks into OpenSearch")


if __name__ == "__main__":
    main()