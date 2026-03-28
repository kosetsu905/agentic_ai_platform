import os
import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from opensearchpy import OpenSearch, helpers


def load_pdf(path):
    docs = []
    pdf = fitz.open(path)

    for page_num, page in enumerate(pdf):
        text = page.get_text()
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path,
                        "page": page_num
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
        print(f"Deleting existing index '{index_name}'...")
        client.indices.delete(index=index_name)

    if not client.indices.exists(index=index_name):
        mapping = {
            "settings": {"index": {"knn": True, "number_of_shards": 3, "number_of_replicas": 1}},
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "metadata": {"type": "object"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 384,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {"ef_construction": 512, "m": 16}
                        }
                    }
                }
            }
        }
        client.indices.create(index=index_name, body=mapping)

    actions = []
    for d in docs:
        vector = embeddings.embed_query(d.page_content)
        actions.append({
            "_index": index_name,
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