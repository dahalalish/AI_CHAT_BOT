import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
import shutil

os.makedirs("vectorstore", exist_ok=True)

df = pd.read_excel("data/payers.xlsx")

df = df.fillna("").drop_duplicates()

docs = []

for _, row in df.iterrows():

    text = f"""
Payer: {row['Payer']}
Dataset Type: {row['Dataset_Type']}
CDF Field: {row['CDF_Field']}
Business Logic: {row['Business_Logic']}
Category: {row.get('Category', '')}
Description: {row.get('Description', '')}
Synonyms: {row.get('Synonyms', '')}
"""

    docs.append(
        Document(
            page_content=text,
            metadata={
                "payer": row["Payer"],
                "cdf_field": row["CDF_Field"]
            }
        )
    )

# Optional: clear old vectorstore before rebuild
if os.path.exists("vectorstore"):
    shutil.rmtree("vectorstore")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="vectorstore"
)

db.persist()

print("✅ Chroma Vector DB created")
