import os
from langchain_community.vectorstores import Chroma
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

Answer ONLY based on the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {"context": retriever | format_docs, "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":
    while True:
        q = input(">> ")
        if q.lower() == "exit":
            break

        print("\nAnswer:\n", chain.invoke(q))

# What is hypertension?
# What are the strategies for hypertension control?
# What are the barriers to hypertension control?