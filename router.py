from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="llama3", temperature=0)

def route_query(query: str):

    prompt = f"""
    Classify query:
    SQL → filtering, counting, structured
    RAG → explanation
    HYBRID → both

    Query: {query}

    Answer ONLY: SQL, RAG, HYBRID
    """

    try:
        decision = llm.invoke(prompt).content.strip().upper()
    except:
        return "RAG"

    if decision not in ["SQL", "RAG", "HYBRID"]:
        return "RAG"

    return decision