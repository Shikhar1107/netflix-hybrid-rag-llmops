from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
from src.retrieval.retriever import retrieve_documents

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

        block = f"""
        Result {index}
        Title: {metadata.get("title", "Unknown")}
        Genre: {metadata.get("genre", "unknown")}
        Country: {metadata.get("country", "unknown")}
        Release Year: {metadata.get("year", "Unknown")}
        Rating: {metadata.get("rating", "Unknown")}
        Duration: {metadata.get("duration", "Unknown")}
        Reranker Score: {score:.4f}
        Content:
        {doc.page_content}
        """.strip()

        context_blocks.append(block)
    
    return "\n\n---\n\n".join(context_blocks)

def build_scores(retrieved_items):
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

    Final answer:
    """.strip(),
    )

def generate_answer(question:str):
    retrieval_output = retrieve_documents(question)
    retrieved_items = retrieval_output["documents"]
    context = format_context(retrieved_items)
    sources = build_scores(retrieved_items)

    reference_title = retrieval_output.get("reference_title") or "None"

    prompt = RAG_PROMPT.format(
        question=question,
        context=context,
        reference_title=reference_title
    )

    llm = get_llm()

    response = llm.invoke(prompt)

    return {
        "question": question,
        "answer": response.content,
        "reference_title": reference_title,
        "filters": retrieval_output.get("filters", {}),
        "sources": sources
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


