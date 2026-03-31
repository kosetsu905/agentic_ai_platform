import os
from opensearchpy import OpenSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import get_tracer
from dotenv import load_dotenv
import pandas as pd
from phoenix.evals.llm import LLM
from phoenix.evals.metrics import CorrectnessEvaluator
from phoenix.evals import bind_evaluator, evaluate_dataframe

load_dotenv()

tracer = get_tracer(__name__)
session = px.launch_app()
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006/v1/traces"
provider = TracerProvider()
processor = SimpleSpanProcessor(OTLPSpanExporter(endpoint=os.environ["PHOENIX_COLLECTOR_ENDPOINT"]))
provider.add_span_processor(processor)
LangChainInstrumentor().instrument(tracer_provider=provider)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}], use_ssl=False)
INDEX_NAME = "medical_docs"
TOP_K = 6

def search(query, top_k=TOP_K):
    q_vector = embeddings.embed_query(query)
    body = {
        "size": top_k,
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "content": query  # keyword search
                        }
                    },
                    {
                        "knn": {
                            "embedding": {
                                "vector": q_vector,
                                "k": top_k
                            }
                        }
                    }
                ],
                "filter": [
                    {
                        "exists": {"field": "metadata.source"}
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

judge_llm = LLM(provider="openai", model="gpt-5.4-nano", client="openai")
correctness_eval = CorrectnessEvaluator(llm=judge_llm)

def evaluate_answer(query, answer):
    df = pd.DataFrame([{
        "attributes.input.value": query,
        "attributes.output.value": answer
    }])

    bound_eval = bind_evaluator(
        evaluator=correctness_eval,
        input_mapping={
            "input": "attributes.input.value",
            "output": "attributes.output.value",
        }
    )

    results_df = evaluate_dataframe(
        dataframe=df,
        evaluators=[bound_eval]
    )
    return results_df

if __name__ == "__main__":
    while True:
        q = input(">> ")
        if q.lower() == "exit":
            break

        with tracer.start_as_current_span("rag_pipeline"):
            with tracer.start_as_current_span("retrieval"):
                docs = search(q)

            with tracer.start_as_current_span("rerank"):
                docs = rerank_docs(q, docs, top_k=3)

            with tracer.start_as_current_span("generation"):
                context = format_docs_for_llm(docs)
                answer = (prompt | llm | StrOutputParser()).invoke({
                    "context": context,
                    "question": q
                })

        sources = extract_sources(docs)

        print("\nAnswer:\n", answer)
        print("\nSources:\n", sources)

        eval_results = evaluate_answer(q, answer)
        result = eval_results["correctness_score"].iloc[0]

        print("\n=== Phoenix Eval Results ===")
        print("Score:", result["score"])
        print("Label:", result["label"])
        print("Explanation:", result["explanation"])

# What is hypertension?
# What are the strategies for hypertension control?
# What are the barriers to hypertension control?
# What is high blood pressure?
# What is a flu?
# What is the lung problem?
# Can you list some types of cancer?
# What cause a oral health problem?
# How head and neck cancer affect mouth health?