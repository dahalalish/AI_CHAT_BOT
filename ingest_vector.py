import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
import shutil

# ==============================
# 1. Reset vectorstore
# ==============================
if os.path.exists("vectorstore"):
    shutil.rmtree("vectorstore")
os.makedirs("vectorstore", exist_ok=True)

# ==============================
# 2. Load Excel (MULTI-SHEET)
# ==============================
file_path = "data/payers.xlsx"

df_mapping = pd.read_excel(file_path, sheet_name="Data_Mapping")
df_metadata = pd.read_excel(file_path, sheet_name="Metadata")

# ==============================
# 3. Clean Data
# ==============================
df_mapping = df_mapping.fillna("").drop_duplicates()
df_metadata = df_metadata.fillna("").drop_duplicates()

# Normalize column names
df_mapping.columns = df_mapping.columns.str.strip().str.upper()
df_metadata.columns = df_metadata.columns.str.strip().str.upper()

# ==============================
# 4. Validate Required Columns
# ==============================
required_cols = [
    "DATASET_TYPE", "PAYER", "VHLAYOUTID",
    "SOURCE", "CDF_FIELD", "BUSINESS_LOGIC"
]

missing_cols = [col for col in required_cols if col not in df_mapping.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in Data_Mapping sheet: {missing_cols}")

# ==============================
# 5. Prepare Metadata Lookup
# ==============================
# Assuming metadata keyed by CDF_FIELD (most common case)
metadata_lookup = {}

if "CDF_FIELD" in df_metadata.columns:
    for _, row in df_metadata.iterrows():
        key = str(row.get("CDF_FIELD", "")).strip()
        metadata_lookup[key] = row.to_dict()

# ==============================
# 6. Build Documents
# ==============================
docs = []

for _, row in df_mapping.iterrows():
    dataset = str(row.get("DATASET_TYPE", "")).strip()
    payer = str(row.get("PAYER", "")).strip()
    vhid = str(row.get("VHLAYOUTID", "")).strip()
    source = str(row.get("SOURCE", "")).strip()
    field = str(row.get("CDF_FIELD", "")).strip()
    logic = str(row.get("BUSINESS_LOGIC", "")).strip()

    # 🔹 Fetch metadata (if exists)
    meta_info = metadata_lookup.get(field, {})

    # Convert metadata dict to readable text
    meta_text = ""
    if meta_info:
        meta_text = "\n".join(
            [f"{k}: {v}" for k, v in meta_info.items() if v]
        )

    # ==============================
    # 🔥 Embedding Text (Optimized)
    # ==============================
    text = f"""
PAYER: {payer}

DATASET TYPE: {dataset}

VHLAYOUTID: {vhid}

SOURCE FIELD: {source}

STANDARD FIELD (CDF): {field}

TRANSFORMATION LOGIC:
{logic}

METADATA:
{meta_text}

CONTEXT:
This record defines payer-specific transformation logic mapping a source field
into a standardized healthcare CDF field.
"""

    # ==============================
    # 🔥 Metadata for Vector DB
    # ==============================
    metadata = {
        "payer": payer,
        "dataset_type": dataset,
        "vh_layout_id": vhid,
        "source_field": source,
        "cdf_field": field
    }

    docs.append(
        Document(
            page_content=text.strip(),
            metadata=metadata
        )
    )

# ==============================
# 7. Embeddings
# ==============================
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ==============================
# 8. Create Vector DB
# ==============================
db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="vectorstore"
)

db.persist()

print(f"Vector DB created successfully")
