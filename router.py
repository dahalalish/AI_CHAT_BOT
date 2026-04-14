from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="llama3", temperature=0)


def route_query(query: str):

    prompt = f"""
You are a strict query classifier for a payer data assistant.

Classification Rules:

SQL:
- Queries asking to list, count, filter, or retrieve structured data
- Example: "list payers using member_id"

RAG:
- Definitions of fields (VERY IMPORTANT)
- Business logic explanation
- Meaning of columns (e.g., memid, plan_id, subscriber_id)
- Any domain-related explanation

HYBRID:
- Queries needing both data + explanation

OUT_OF_SCOPE:
- Completely unrelated to payer data
- Personal chat (e.g., "how are you", "who are you")

IMPORTANT:
- If a query asks about a FIELD NAME (like memid, member_id, plan_id),
  ALWAYS classify as RAG
- NEVER mark domain-related queries as OUT_OF_SCOPE

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
