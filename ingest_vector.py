import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os

os.makedirs("vectorstore", exist_ok=True)

df = pd.read_excel("data/payers.xlsx")

docs = []
for _, row in df.iterrows():
    text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
    docs.append(text)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.create_documents(docs)

embeddings = OllamaEmbeddings(model="llama3")

db = FAISS.from_documents(chunks, embeddings)
db.save_local("vectorstore")

print("✅ Vector DB created")