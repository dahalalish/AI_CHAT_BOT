from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.chat_models import ChatOllama


def get_sql_chain():

    db = SQLDatabase.from_uri("sqlite:///db/payers.db")

    llm = ChatOllama(
        model="llama3",
        temperature=0
    )

    chain = SQLDatabaseChain.from_llm(
        llm=llm,
        db=db,
        verbose=False,
        return_intermediate_steps=True
    )

    return chain
