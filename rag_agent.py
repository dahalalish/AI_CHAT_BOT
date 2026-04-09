from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


def get_rag_chain():
    embeddings = OllamaEmbeddings(model="llama3")

    db = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatOllama(model="llama3", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
Answer the user's question using only the provided context.

Context:
{context}

Question:
{input}
""")

    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    rag_chain = create_retrieval_chain(
        retriever,
        combine_docs_chain
    )

    return rag_chain
