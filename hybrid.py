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

    # SQL
    if route == "SQL":
        return {"type": "SQL", "answer": sql_chain.run(query)}

    # RAG
    if route == "RAG":
        rag_result = rag_chain.invoke({"input": query})
        return {"type": "RAG", "answer": rag_result["answer"]}


    # HYBRID
    if route == "HYBRID":
        with concurrent.futures.ThreadPoolExecutor() as executor:
            sql_future = executor.submit(sql_chain.run, query)
            rag_future = executor.submit(rag_chain.invoke, {"input": query})

            sql_result = sql_future.result()
            rag_result = rag_future.result()["answer"]

        final_prompt = f"""
        Combine results:

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
            "rag": rag_result
        }