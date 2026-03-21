import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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

retriever = db.as_retriever(search_kwargs={"k": 5})

llm = OllamaLLM(model="llama3:latest")

prompt = ChatPromptTemplate.from_template("""
You are a medical assistant.

STRICT RULES:
- Use ONLY the provided context
- Do NOT use your own knowledge
- If the answer is not explicitly in the context, say "I don't know"
- DO NOT repeat source references in the answer

Context:
{context}

Question:
{question}

Answer:
""")

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

        context = format_docs(docs)
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