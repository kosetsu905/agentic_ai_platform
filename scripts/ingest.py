# =============================================================================
# ingest.py - Medical PDF Vectorization & OpenSearch Indexing Pipeline
# =============================================================================
# Dependencies:
#   pip install pymupdf langchain-core langchain-text-splitters langchain-huggingface opensearch-py tqdm
#
# .env Example:
#   DATA_DIR=./data
#   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
#   OPENSEARCH_HOST=localhost
#   OPENSEARCH_PORT=9200
#   OPENSEARCH_INDEX=medical_docs
#   OPENSEARCH_USER=admin
#   OPENSEARCH_PASSWORD=your_password
#   OPENSEARCH_USE_SSL=false
#   FORCE_REINDEX=false
#   CHUNK_SIZE=500
#   CHUNK_OVERLAP=100
#   BATCH_SIZE=64
# =============================================================================

import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from opensearchpy import OpenSearch, helpers
from tqdm import tqdm

# ---- Logging Setup ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# ---- Configuration (env vars with defaults) ----
DATA_DIR = Path(os.getenv('DATA_DIR', './data'))
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
OPENSEARCH_HOST = os.getenv('OPENSEARCH_HOST', 'localhost')
OPENSEARCH_PORT = int(os.getenv('OPENSEARCH_PORT', '9200'))
OPENSEARCH_INDEX = os.getenv('OPENSEARCH_INDEX', 'medical_docs')
OPENSEARCH_USER = os.getenv('OPENSEARCH_USER', 'admin')
OPENSEARCH_PASSWORD = os.getenv('OPENSEARCH_PASSWORD', '')
OPENSEARCH_USE_SSL = os.getenv('OPENSEARCH_USE_SSL', 'false').lower() == 'true'
FORCE_REINDEX = os.getenv('FORCE_REINDEX', 'false').lower() == 'true'
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '100'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '64'))
EXPECTED_DIM = 384


def load_pdfs_from_dir(data_dir: Path) -> List[Document]:
    """Load all PDF files from a directory and convert each page to a Document."""
    documents: List[Document] = []
    pdf_files = sorted(data_dir.glob('*.pdf'))

    if not pdf_files:
        logger.warning('No PDF files found in %s', data_dir.resolve())

    for pdf_path in pdf_files:
        try:
            pdf = fitz.open(pdf_path)
        except Exception as exc:
            logger.warning('Skipping unreadable PDF %s: %s', pdf_path.name, exc)
            continue

        filename = pdf_path.name
        for page_num, page in enumerate(pdf):
            text = page.get_text()
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            'source': filename,
                            'page': page_num,
                            'doc_type': 'medical',
                        },
                    )
                )
        page_count = pdf.page_count
        pdf.close()
        logger.info('Loaded %d pages from %s', page_count, filename)

    logger.info('Total documents loaded: %d', len(documents))
    return documents


def split_documents(documents: List[Document], chunk_size: int = CHUNK_SIZE,
                    chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = splitter.split_documents(documents)
    logger.info('Split into %d chunks (chunk_size=%d, overlap=%d)', len(docs), chunk_size, chunk_overlap)
    return docs


def _build_index_mapping(index_name: str) -> dict:
    """Build OpenSearch index mapping body.

    NOTE: space_type uses 'cosine' (not 'cosinesimil') per OpenSearch specification.
    """
    return {
        'settings': {
            'index': {
                'knn': True,
                'number_of_shards': 3,
                'number_of_replicas': 1,
            }
        },
        'mappings': {
            'properties': {
                'content': {'type': 'text'},
                'metadata': {
                    'properties': {
                        'source': {'type': 'keyword'},
                        'page': {'type': 'integer'},
                        'doc_type': {'type': 'keyword'},
                    }
                },
                'embedding': {
                    'type': 'knn_vector',
                    'dimension': EXPECTED_DIM,
                    'method': {
                        'name': 'hnsw',
                        'engine': 'faiss',
                        'space_type': 'cosine',
                    },
                },
            }
        },
    }


def create_index_if_needed(client: OpenSearch, index_name: str) -> None:
    """Create the index if it does not exist.

    If FORCE_REINDEX is true, delete and recreate on every run.
    Otherwise, skip creation if the index already exists (append mode).
    """
    exists = client.indices.exists(index=index_name)

    if exists and FORCE_REINDEX:
        logger.info('FORCE_REINDEX=true, deleting existing index %s', index_name)
        try:
            client.indices.delete(index=index_name)
        except Exception as exc:
            logger.error('Failed to delete index %s: %s', index_name, exc)
            raise
        exists = False

    if not exists:
        mapping = _build_index_mapping(index_name)
        try:
            client.indices.create(index=index_name, body=mapping)
            logger.info('Index %s created successfully', index_name)
        except Exception as exc:
            logger.error('Failed to create index %s: %s', index_name, exc)
            raise
    else:
        logger.info('Index %s already exists, appending documents', index_name)


def _build_bulk_actions(
    batch_docs: List[Document],
    batch_vectors: List[List[float]],
    index_name: str,
) -> Generator[dict, None, None]:
    """Generator that yields bulk-index actions for a single batch."""
    for doc, vector in zip(batch_docs, batch_vectors):
        yield {
            '_index': index_name,
            '_id': str(uuid.uuid4()),
            '_source': {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'embedding': vector,
            },
        }


def batch_embed_and_index(
    docs: List[Document],
    embeddings: HuggingFaceEmbeddings,
    client: OpenSearch,
    index_name: str,
    batch_size: int = BATCH_SIZE,
) -> None:
    """Batch-embed documents and index into OpenSearch using streaming bulk writes."""
    total = len(docs)
    indexed = 0

    pbar = tqdm(total=total, desc='Embedding & indexing', unit='chunk')

    for start in range(0, total, batch_size):
        batch_docs = docs[start:start + batch_size]
        texts = [d.page_content for d in batch_docs]

        try:
            vectors = embeddings.embed_documents(texts)
        except Exception as exc:
            logger.error('Embedding failed for batch starting at %d: %s', start, exc)
            pbar.update(len(batch_docs))
            continue

        # Dimension validation
        valid_docs = []
        valid_vectors = []
        for doc, vector in zip(batch_docs, vectors):
            if len(vector) != EXPECTED_DIM:
                logger.error(
                    'Dimension mismatch: expected %d, got %d. Skipping chunk from %s page %s',
                    EXPECTED_DIM, len(vector), doc.metadata.get('source'), doc.metadata.get('page'),
                )
                continue
            valid_docs.append(doc)
            valid_vectors.append(vector)

        if not valid_docs:
            pbar.update(len(batch_docs))
            continue

        actions = _build_bulk_actions(valid_docs, valid_vectors, index_name)

        try:
            success, errors = helpers.bulk(client, actions, stats_only=False, raise_on_error=False)
            if errors:
                logger.warning('Bulk write had %d errors in batch starting at %d', len(errors), start)
            indexed += success
        except Exception as exc:
            logger.error('Bulk write failed for batch starting at %d: %s', start, exc)

        pbar.update(len(batch_docs))

    pbar.close()
    logger.info('Successfully indexed %d/%d chunks into %s', indexed, total, index_name)


# ---- Entry Point ----
def main() -> None:
    """Main pipeline: load PDFs, split, embed, and index into OpenSearch."""
    logger.info('===== Medical PDF Ingest Pipeline Starting =====')

    # 1. Load PDFs
    documents = load_pdfs_from_dir(DATA_DIR)
    if not documents:
        logger.warning('No documents to process. Exiting.')
        return

    # 2. Split into chunks
    docs = split_documents(documents)

    # 3. Initialize embedding model
    logger.info('Loading embedding model: %s', EMBEDDING_MODEL)
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception as exc:
        logger.error('Failed to load embedding model %s: %s', EMBEDDING_MODEL, exc)
        sys.exit(1)

    # 4. Connect to OpenSearch
    use_ssl = OPENSEARCH_USE_SSL
    opensearch_kwargs: dict = {
        'hosts': [{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
        'use_ssl': use_ssl,
        'verify_certs': use_ssl,
    }
    if OPENSEARCH_USER and OPENSEARCH_PASSWORD:
        opensearch_kwargs['http_auth'] = (OPENSEARCH_USER, OPENSEARCH_PASSWORD)

    try:
        client = OpenSearch(**opensearch_kwargs)
        info = client.info()
        logger.info('Connected to OpenSearch %s', info.get('version', {}).get('number', 'unknown'))
    except Exception as exc:
        logger.error('Failed to connect to OpenSearch at %s:%d: %s', OPENSEARCH_HOST, OPENSEARCH_PORT, exc)
        sys.exit(1)

    # 5. Create / verify index
    try:
        create_index_if_needed(client, OPENSEARCH_INDEX)
    except Exception as exc:
        logger.error('Index setup failed: %s', exc)
        sys.exit(1)

    # 6. Batch embed & index
    batch_embed_and_index(docs, embeddings, client, OPENSEARCH_INDEX, BATCH_SIZE)

    logger.info('===== Ingest Pipeline Complete =====')


if __name__ == '__main__':
    main()
