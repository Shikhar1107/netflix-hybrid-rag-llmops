from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
from src.retrieval.hybrid_retriever import retrieve_documents
import time
load_dotenv()

def format_context(retrieved_items):
    """
    Converts retrieved documents into clean context for the LLM
    """

    context_blocks = []

    for index, item in enumerate(retrieved_items, start = 1):
        doc = item["document"]
        score = item["score"]
        metadata = doc.metadata
        description = doc.page_content.split("Description:")[-1].strip()
        block = f"""
        Result {index}
        Title: {metadata.get("title", "Unknown")}
        Genre: {metadata.get("genre", "unknown")}
        Country: {metadata.get("country", "unknown")}
        Release Year: {metadata.get("year", "Unknown")}
        Rating: {metadata.get("rating", "Unknown")}
        Duration: {metadata.get("duration", "Unknown")}
        Reranker Score: {score:.4f}
        Description:
        {description}
        """.strip()

        context_blocks.append(block)
    
    return "\n\n---\n\n".join(context_blocks)

def build_sources(retrieved_items):
    sources = []
    
    for item in retrieved_items:
        doc = item["document"]
        metadata= doc.metadata

        sources.append({
            "title": metadata.get("title", "Unknown"),
            "type": metadata.get("type", "Unknown"),
            "year": metadata.get("year", "Unknown"),
            "rating": metadata.get("rating", "Unknown"),
            "country": metadata.get("country", "Unknown"),
            "genre": metadata.get("genre", "Unknown"),
            "score": round(float(item["score"]), 4),
        })
    
    return sources

def get_llm():
    return ChatOpenAI(
        model="openai/gpt-oss-20b:free",
        temperature=0.3,
        base_url = "https://openrouter.ai/api/v1",
        api_key=os.getenv('OPENROUTER_API_KEY')
    )

RAG_PROMPT =PromptTemplate(
    input_variables=["question","context","reference_title"],
    template="""
    You are a Netflix content recommendation assistant.

    Use ONLY the provided context to answer the user's question.
    Do not invent titles that are not present in the context.
    Do not recommend the reference title itself.
    If the context is weak, say that the match is approximate.
    Do not claim a title is a Netflix Original unless the context explicitly says so.
    Use only facts explicitly present in the retrieved context.
    Do not infer themes unless the description or genre supports them.
    Do not mention “time-bending,” “mind-bending,” or “supernatural” unless those words or equivalent concepts appear in the context.
    Keep explanations short and evidence-based.
    Copy metadata exactly from context.
    Recommend every title provided in the retrieved context unless the title is the reference title itself.
    The "Sources used" list must exactly match the recommended titles.
    Do not omit retrieved titles from the final answer.

    Reference title, if any:
    {reference_title}

    User question:
    {question}

    Retrieved Netflix context:
    {context}

    Answer requirements:
    - Give a short intro sentence.
    - Recommend up to 5 titles.
    - For each title, explain why it matches the user's request.
    - Mention type, year, country, rating, and genre where useful.
    - Keep the answer concise and helpful.
    - End with a "Sources used" list containing only the recommended titles.
    - Copy rating, year, country, type, and genre exactly from the provided context. Do not rewrite or normalize metadata values.
    - Recommend only titles that are present in the retrieved context and sources list.

    Final answer:
    """.strip(),
    )

def build_eval_contexts(retrieved_items):
    eval_contexts = []

    for item in retrieved_items:
        doc = item["document"]
        metadata = doc.metadata

        description = doc.page_content.split("Description:")[-1].strip()

        context = f"""
Title: {metadata.get("title", "Unknown")}
Type: {metadata.get("type", "Unknown")}
Genre: {metadata.get("genre", "Unknown")}
Country: {metadata.get("country", "Unknown")}
Release Year: {metadata.get("year", "Unknown")}
Rating: {metadata.get("rating", "Unknown")}
Duration: {metadata.get("duration", "Unknown")}
Description: {description}
""".strip()

        eval_contexts.append(context)

    return eval_contexts

def generate_answer(question: str, retrieval_mode: str = "auto"):
    """
    Full RAG pipeline:
    user question -> retrieval -> context -> LLM answer.
    """

    timings = {}

    retrieval_start = time.time()
    retrieval_output = retrieve_documents(
    query=question,
    retrieval_mode=retrieval_mode,
    )
    timings["retrieval_seconds"] = round(time.time() - retrieval_start, 3)

    context_start = time.time()
    retrieved_items = retrieval_output["documents"]
    context = format_context(retrieved_items)
    sources = build_sources(retrieved_items)
    eval_contexts = build_eval_contexts(retrieved_items)

    reference_title = retrieval_output.get("reference_title") or "None"

    prompt = RAG_PROMPT.format(
        question=question,
        context=context,
        reference_title=reference_title,
    )

    timings["context_building_seconds"] = round(time.time() - context_start, 3)

    llm_start = time.time()
    llm = get_llm()
    response = llm.invoke(prompt)
    timings["llm_generation_seconds"] = round(time.time() - llm_start, 3)

    return {
        "question": question,
        "answer": response.content,
        "reference_title": reference_title,
        "filters": retrieval_output.get("filters", {}),
        "sources": sources,
        "timings": timings,
        "contexts": eval_contexts,
        "retrieval_mode": retrieval_output.get("retrieval_mode"),
        "vector_count": retrieval_output.get("vector_count", 0),
        "graph_count": retrieval_output.get("graph_count", 0),
        "final_count": retrieval_output.get("final_count", 0),
    }

if __name__ == "__main__":
    result = generate_answer("thriller shows like Dark")

    print("\nQusetion:")
    print(result["question"])

    print("\nReference title:")
    print(result["reference_title"])

    print("\nFilters:")
    print(result["filters"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print(source)


