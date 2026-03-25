import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

print("DB PATH:", DB_DIR)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 6})

llm = OllamaLLM(model="llama3:latest")

prompt = ChatPromptTemplate.from_template("""
You are a medical assistant.

STRICT RULES:
- Answer ONLY using the provided context
- If the answer is not clearly in the context, say "I don't know"
- Be concise and factual
- Do NOT make assumptions

Context:
{context}

Question:
{question}

Answer:
""")

def rerank(query, docs, top_k=3):
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs, batch_size=8)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

def format_docs_for_llm(docs, max_chars=2000):
    text = ""
    for d in docs:
        chunk = d.page_content.strip()
        if len(text) + len(chunk) > max_chars:
            break
        text += chunk + "\n\n"
    return text


def format_docs(docs):
    formatted = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "unknown")

        formatted.append(
            f"[Source: {os.path.basename(source)} | Page: {page}]\n{d.page_content}"
        )

    return "\n\n---\n\n".join(formatted)


chain = (
    {"context": retriever | format_docs, "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)


def extract_sources(docs):
    sources = set()
    for d in docs:
        src = os.path.basename(d.metadata.get("source", "unknown"))
        page = d.metadata.get("page", "unknown")
        sources.add(f"{src} (Page {page})")
    return "\n".join(f"- {s}" for s in sources)


if __name__ == "__main__":
    while True:
        q = input(">> ")
        if q.lower() == "exit":
            break

        docs = retriever.invoke(q)
        docs = rerank(q, docs, top_k=3)

        context = format_docs_for_llm(docs)
        answer = (prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "question": q
        })

        sources = extract_sources(docs)

        print("\nAnswer:\n", answer)
        print("\nSources:\n", sources)

# What is hypertension?
# What are the strategies for hypertension control?
# What are the barriers to hypertension control?
# What is high blood pressure?
# What is a flu?
# What is the lung problem?
# Can you list some types of cancer?
# What cause a oral health problem?
# How head and neck cancer affect mouth health?