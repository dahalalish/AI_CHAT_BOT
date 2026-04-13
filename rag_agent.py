from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


def get_rag_chain():

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 15, "fetch_k": 30}
    )

    llm = ChatOllama(model="llama3", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
Answer using ONLY the context.

IMPORTANT:
- Return ALL matching payers
- Do NOT omit records

Context:
{context}

Question:
{input}
""")

    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    return create_retrieval_chain(retriever, combine_docs_chain)
