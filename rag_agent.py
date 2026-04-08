from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.chat_models import ChatOllama

def get_rag_chain():
    embeddings = OllamaEmbeddings(model="llama3")

    db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatOllama(model="llama3", temperature=0)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa