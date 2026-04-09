import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os

os.makedirs("vectorstore", exist_ok=True)

df = pd.read_excel("data/payers.xlsx")

docs = []
for _, row in df.iterrows():
    text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
    docs.append(text)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.create_documents(docs)

embeddings = OllamaEmbeddings(model="llama3")

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="vectorstore"
)

print("✅ Chroma Vector DB created")
