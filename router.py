from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="llama3", temperature=0)


def route_query(query: str):

    prompt = f"""
Classify the user query.

SQL:
- filtering
- counting
- structured tabular lookup
- listing payers

RAG:
- explanations
- business logic interpretation

HYBRID:
- needs both SQL and explanation

OUT_OF_SCOPE:
- unrelated/general chat
- personal questions
- greetings

Query:
{query}

Return ONLY:
SQL / RAG / HYBRID / OUT_OF_SCOPE
"""

    try:
        decision = llm.invoke(prompt).content.strip().upper()
    except:
        return "RAG"

    if decision not in ["SQL", "RAG", "HYBRID", "OUT_OF_SCOPE"]:
        return "RAG"

    return decision
