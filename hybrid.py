from sql_agent import get_sql_chain
from rag_agent import get_rag_chain
from router import route_query
from intent_handler import (
    detect_intent,
    generate_smalltalk_response,
    generate_toxic_response,
    generate_out_of_scope_response
)
from langchain_community.chat_models import ChatOllama
import concurrent.futures

sql_chain = get_sql_chain()
rag_chain = get_rag_chain()
llm = ChatOllama(model="llama3", temperature=0)


def hybrid_execute(query: str):

    # INTENT HANDLING
    intent = detect_intent(query)

    if intent == "CLEAR":
        return {"type": "CLEAR", "answer": "Chat cleared."}

    if intent == "SMALL_TALK":
        return {"type": "SMALL_TALK", "answer": generate_smalltalk_response()}

    if intent == "TOXIC":
        return {"type": "TOXIC", "answer": generate_toxic_response()}

    # ROUTING
    route = route_query(query)

    if route == "OUT_OF_SCOPE":
    # fallback to RAG for short domain-like queries
        if len(query.split()) <= 5:
            rag_result = rag_chain.invoke({"input": query})
            return {
                "type": "RAG_FALLBACK",
                "answer": rag_result["answer"]
            }

        return {
            "type": "OUT_OF_SCOPE",
            "answer": generate_out_of_scope_response()
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
            sql_future = executor.submit(sql_chain.invoke, {"query": query})
            rag_future = executor.submit(rag_chain.invoke, {"input": query})

            sql_output = sql_future.result()
            rag_output = rag_future.result()

            sql_result = sql_output["result"]
            rag_result = rag_output["answer"]

        final_prompt = f"""
Combine results carefully.

SQL:
{sql_result}

RAG:
{rag_result}

Return complete answer without omitting records.
"""

        final = llm.invoke(final_prompt).content

        return {
            "type": "HYBRID",
            "answer": final,
            "sql": sql_result,
            "rag": rag_result,
            "sql_steps": sql_output["intermediate_steps"]
        }