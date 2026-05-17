import os
import logging
from opensearchpy import OpenSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# === Context control parameters ===
CONTEXT_MAX_TURNS = int(os.getenv("CONTEXT_MAX_TURNS", "5"))
CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "2000"))
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "true").lower() == "true"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}], use_ssl=False)
INDEX_NAME = "medical_docs"
TOP_K = 10
PIPELINE_NAME = "hybrid-search-pipeline"
pipeline_body = {
    "description": "Hybrid search pipeline with RRF",
    "phase_results_processors": [
        {
            "normalization-processor": {
                "normalization": {
                    "technique": "min_max"
                },
                "combination": {
                    "technique": "rrf",
                    "parameters": {
                        "rank_constant": 60
                    }
                }
            }
        }
    ]
}

def init_pipeline():
    try:
        client.transport.perform_request(
            "GET",
            f"/_search/pipeline/{PIPELINE_NAME}"
        )
        logger.info("Pipeline %s already exists.", PIPELINE_NAME)
    except Exception:
        try:
            client.transport.perform_request(
                "PUT",
                f"/_search/pipeline/{PIPELINE_NAME}",
                body=pipeline_body
            )
            logger.info("Pipeline %s created successfully.", PIPELINE_NAME)
            client.indices.put_settings(
                index=INDEX_NAME,
                body={
                    "index.search.default_pipeline": PIPELINE_NAME
                }
            )
        except Exception as e:
            logger.error("Failed to create pipeline: %s", e)

init_pipeline()

def search(query, top_k=TOP_K):
    q_vector = embeddings.embed_query(query)
    knn_k = top_k * 10

    body = {
        "size": top_k,
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "match": {
                            "content": query
                        }
                    },
                    {
                        "knn": {
                            "embedding": {
                                "vector": q_vector,
                                "k": knn_k,
                                "filter": {
                                    "term": {
                                        "metadata.doc_type": "medical"
                                    }
                                }
                            }
                        }
                    }
                ]
            }
        }
    }

    res = client.search(index=INDEX_NAME, body=body)

    docs = []
    for hit in res["hits"]["hits"]:
        source = hit["_source"]
        docs.append({
            "content": source["content"],
            "metadata": source["metadata"]
        })

    return docs

def rerank_docs(query, docs, top_k=3):
    pairs = [(query, d["content"]) for d in docs]
    scores = reranker.predict(pairs, batch_size=8)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

def format_docs_for_llm(docs, max_chars=2000):
    text = ""
    for d in docs:
        chunk = d["content"].strip()
        if len(text) + len(chunk) > max_chars:
            break
        text += chunk + "\n\n"
    return text

def extract_sources(docs):
    sources = set()
    for d in docs:
        src = os.path.basename(d["metadata"].get("source", "unknown"))
        page = d["metadata"].get("page", "unknown")
        sources.add(f"{src} (Page {page})")
    return "\n".join(f"- {s}" for s in sources)

# === Chat history handling ===

def format_history(history: list[dict]) -> str:
    """Format chat history into LLM-readable text."""
    if not history:
        return "None"
    lines = []
    for m in history:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines)


def truncate_history(
    history: list[dict] | None,
    max_turns: int = CONTEXT_MAX_TURNS,
    max_chars: int = CONTEXT_MAX_CHARS,
) -> list[dict]:
    """
    Apply dual truncation to chat history:
    1. Turn truncation: keep the most recent max_turns rounds (each round = user + assistant).
    2. Char truncation: limit the formatted history text to max_chars,
       preserving recent messages from the tail to stay within the LLM context window.
    """
    if not history:
        return []

    # Step 1: turn truncation
    max_messages = max_turns * 2
    history = history[-max_messages:]

    # Step 2: char truncation (keep from tail, discard earliest messages)
    while True:
        text = format_history(history)
        if len(text) <= max_chars or len(history) <= 2:
            break
        # Discard the earliest round (user + assistant)
        history = history[2:]

    return history


def rewrite_query_with_context(q: str, history: list[dict]) -> str:
    """Use LLM to rewrite a context-dependent question into a self-contained query."""
    if not history or not ENABLE_QUERY_REWRITE:
        return q

    rewrite_prompt = ChatPromptTemplate.from_template("""
Given the conversation history and the user's latest question, rewrite the question
so that it is self-contained and does not rely on pronouns or references from the history.
If the question is already self-contained, return it as-is.

History:
{history}

Question: {question}

Rewritten Question:
""")

    try:
        history_text = format_history(history)
        chain = rewrite_prompt | llm | StrOutputParser()
        rewritten = chain.invoke({"history": history_text, "question": q})
        logger.info("Query rewritten: %s -> %s", q, rewritten.strip())
        return rewritten.strip()
    except Exception as exc:
        logger.warning("Query rewrite failed: %s, falling back to original query", exc)
        return q


llm = OllamaLLM(model="llama3:latest")
prompt = ChatPromptTemplate.from_template("""
You are a medical assistant.

STRICT RULES:
- Answer ONLY using the provided context
- If the answer is not clearly in the context, say "I don't know"
- Be concise and factual
- Do NOT make assumptions
- If there is chat history, use it to understand the context of the question

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
""")

serper = GoogleSerperAPIWrapper()

def web_search(query, k=3):
    results = serper.results(query)

    docs = []

    if "answerBox" in results:
        docs.append(results["answerBox"].get("snippet", ""))

    for r in results.get("organic", [])[:k]:
        docs.append(r.get("snippet", ""))

    return docs

class HybridRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str):
        local_docs = search(query)
        local_docs = rerank_docs(query, local_docs, top_k=3)

        web_docs = web_search(query)
        print(web_docs)

        all_docs = local_docs + [
            {"content": d, "metadata": {"source": "web"}}
            for d in web_docs
        ]

        all_docs = rerank_docs(query, all_docs, top_k=4)

        return [
            Document(
                page_content=d["content"],
                metadata=d["metadata"]
            )
            for d in all_docs
        ]

retriever = HybridRetriever()


def ask_question(q: str, history: list[dict] | None = None):
    """
    Main Q&A function with multi-turn conversation context support.

    Args:
        q: Current user question.
        history: List of past messages in [{"role": "user/assistant", "content": "..."}, ...] format.
                 Falls back to single-turn mode when None or empty.
    """
    history = truncate_history(history)

    # Context-aware query rewrite (triggered only when history is non-empty)
    search_query = rewrite_query_with_context(q, history) if history else q

    # Retrieve (using rewritten query for better recall)
    docs = retriever.invoke(search_query)

    context = "\n\n".join([d.page_content for d in docs])
    chat_history_text = format_history(history)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": q,
        "chat_history": chat_history_text,
    })

    return answer, docs

# What is hypertension?
# What are the strategies for hypertension control?
# What are the barriers to hypertension control?
# What is high blood pressure?
# What is a flu?
# What is the lung problem?
# Can you list some types of cancer?
# What cause a oral health problem?
# How head and neck cancer affect mouth health?
