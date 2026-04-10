from sql_agent import get_sql_chain
from rag_agent import get_rag_chain
from router import route_query
from langchain_community.chat_models import ChatOllama
import concurrent.futures

sql_chain = get_sql_chain()
rag_chain = get_rag_chain()
llm = ChatOllama(model="llama3", temperature=0)


def hybrid_execute(query: str):

    route = route_query(query)

    # OUT OF SCOPE
    if route == "OUT_OF_SCOPE":
        return {
            "type": "OUT_OF_SCOPE",
            "answer": """
I can help only with payer-related questions.

Examples:
• Which payers use member_id?
• Show mapping for Aetna
• Explain business logic for plan_id
"""
        }

    # SQL
    if route == "SQL":

        sql_output = sql_chain.invoke({"query": query})

        return {
            "type": "SQL",
            "answer": sql_output["result"],
            "sql_steps": sql_output["intermediate_steps"]
        }

    # RAG
    if route == "RAG":

        rag_result = rag_chain.invoke({"input": query})

        return {
            "type": "RAG",
            "answer": rag_result["answer"]
        }

    # HYBRID
    if route == "HYBRID":

        with concurrent.futures.ThreadPoolExecutor() as executor:

            sql_future = executor.submit(
                sql_chain.invoke,
                {"query": query}
            )

            rag_future = executor.submit(
                rag_chain.invoke,
                {"input": query}
            )

            sql_output = sql_future.result()
            rag_output = rag_future.result()

            sql_result = sql_output["result"]
            rag_result = rag_output["answer"]

        final_prompt = f"""
Combine results carefully.

IMPORTANT:
- Preserve ALL SQL rows.
- Do NOT omit records.
- Use RAG only for explanation.

SQL:
{sql_result}

RAG:
{rag_result}

Give final clear answer.
"""

        final = llm.invoke(final_prompt).content

        return {
            "type": "HYBRID",
            "answer": final,
            "sql": sql_result,
            "rag": rag_result,
            "sql_steps": sql_output["intermediate_steps"]
        }
