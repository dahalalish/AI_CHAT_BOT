import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import shutil
import hashlib

# ============================================================
# CONFIG
# ============================================================
VECTORSTORE_DIR = "vectorstore"
EXCEL_PATH = "data/payers.xlsx"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

EMBED_MODEL = "nomic-embed-text"

# ============================================================
# HELPERS
# ============================================================

def stable_id(*parts: str) -> str:
    """Deterministic document ID from content parts."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.md5(raw.encode()).hexdigest()


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.fillna("").drop_duplicates()
    df.columns = df.columns.str.strip().str.upper()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def build_splitter(chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )


# ============================================================
# SHEET PROCESSORS
# ============================================================

def process_mapping_sheet(df: pd.DataFrame, sheet: str, metadata_lookup: dict) -> list[Document]:
    """
    Builds rich, chunked documents for every mapping row.

    Each row becomes one semantic document that fuses:
      - payer identity + layout context
      - source → CDF field mapping
      - raw transformation logic (SQL / expressions)
      - related CDF metadata (type, format, description)
      - cross-field hints from the metadata sheet
    """
    splitter = build_splitter()
    docs = []

    for _, row in df.iterrows():
        dataset   = row.get("DATASET_TYPE", "")
        payer     = row.get("PAYER", "")
        vhid      = row.get("VHLAYOUTID", "")
        source    = row.get("SOURCE", "")
        cdf_field = row.get("CDF_FIELD", "")
        logic     = row.get("BUSINESS_LOGIC", "")

        # Pull enriched metadata for this CDF field
        meta = metadata_lookup.get(cdf_field, {})
        data_type   = meta.get("DATA_TYPE", "")
        fmt         = meta.get("FORMAT", "")
        description = meta.get("DESCRIPTION", "")

        # ── Main semantic block ──────────────────────────────
        body = f"""
PAYER: {payer}
DATASET TYPE: {dataset}
VHLAYOUTID: {vhid}
SOURCE FIELD: {source}
STANDARD CDF FIELD: {cdf_field}

TRANSFORMATION LOGIC:
{logic}

CDF FIELD METADATA:
  Data Type   : {data_type}
  Format      : {fmt}
  Description : {description}

CONTEXT:
This record defines how payer {payer} maps the source field '{source}'
into the standardised CDF field '{cdf_field}' for {dataset} data
(VH Layout {vhid}).  The business logic expression above must be
applied verbatim during ETL.
""".strip()

        # ── Optional: secondary chunk focused on logic only ──
        # Helps retrieval when the user asks purely about SQL/transformations.
        logic_block = f"""
PAYER: {payer}  |  DATASET: {dataset}  |  CDF FIELD: {cdf_field}

TRANSFORMATION LOGIC (raw expression):
{logic}

SOURCE FIELD: {source}
CDF DATA TYPE: {data_type}   FORMAT: {fmt}
""".strip()

        base_meta = {
            "sheet": sheet,
            "type": "mapping",
            "payer": payer.lower(),
            "dataset_type": dataset.lower(),
            "vh_layout_id": str(vhid),
            "source_field": source.lower(),
            "cdf_field": cdf_field.lower(),
        }

        # Split the main body (catches long BUSINESS_LOGIC expressions)
        for i, chunk in enumerate(splitter.split_text(body)):
            docs.append(Document(
                page_content=chunk,
                metadata={**base_meta, "chunk_role": "full", "chunk_index": i,
                          "doc_id": stable_id(payer, dataset, cdf_field, "full", str(i))},
            ))

        # Logic-focused chunk (always a single doc — logic rarely exceeds limit)
        if logic:
            docs.append(Document(
                page_content=logic_block,
                metadata={**base_meta, "chunk_role": "logic",
                          "doc_id": stable_id(payer, dataset, cdf_field, "logic")},
            ))

    return docs


def process_metadata_sheet(df: pd.DataFrame, sheet: str) -> tuple[list[Document], dict]:
    """
    Returns:
      - A list of Documents (one per field, split if long)
      - metadata_lookup dict keyed by CDF_FIELD
    """
    splitter = build_splitter(chunk_size=400, overlap=60)
    docs = []
    lookup = {}

    for _, row in df.iterrows():
        field = row.get("CDF_FIELD", "").strip()
        if not field:
            continue

        row_dict = {k: v for k, v in row.to_dict().items() if v and v != "nan"}
        lookup[field] = row_dict

        content = "\n".join(f"{k}: {v}" for k, v in row_dict.items())
        body = f"METADATA RECORD — CDF FIELD: {field}\n{content}"

        for i, chunk in enumerate(splitter.split_text(body)):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "sheet": sheet,
                    "type": "metadata",
                    "cdf_field": field.lower(),
                    "chunk_index": i,
                    "doc_id": stable_id(field, "meta", str(i)),
                },
            ))

    return docs, lookup


def process_cdf_fields_sheet(df: pd.DataFrame, sheet: str) -> list[Document]:
    """
    CDF_Fields sheet contains the canonical field lists per dataset type
    (ELIG, CLAIMS, RXCLAIMS).  We turn each column into a compact document
    so the retriever can answer "what fields exist in CLAIMS / ELIG / RXCLAIMS?"
    """
    docs = []
    for col in df.columns:
        fields = df[col].dropna().astype(str).str.strip()
        fields = [f for f in fields if f and f.lower() != "nan"]
        if not fields:
            continue

        body = f"""
CDF FIELD LIST — DATASET TYPE: {col}

The following are all standard CDF fields for the {col} dataset:
{chr(10).join(f"  - {f}" for f in fields)}

Total fields: {len(fields)}
""".strip()

        docs.append(Document(
            page_content=body,
            metadata={
                "sheet": sheet,
                "type": "cdf_field_list",
                "dataset_type": col.lower(),
                "doc_id": stable_id(col, "cdf_fields"),
            },
        ))

    return docs


def build_payer_summary_docs(df_map: pd.DataFrame, metadata_lookup: dict) -> list[Document]:
    """
    One consolidated summary document per (payer, dataset_type) combination.
    Helps with broad questions like "show everything for Aetna ELIG".
    """
    docs = []
    splitter = build_splitter(chunk_size=800, overlap=150)

    for (payer, dataset), grp in df_map.groupby(["PAYER", "DATASET_TYPE"]):
        vhids = grp["VHLAYOUTID"].unique().tolist()
        lines = []
        for _, row in grp.iterrows():
            src   = row.get("SOURCE", "")
            field = row.get("CDF_FIELD", "")
            logic = row.get("BUSINESS_LOGIC", "")
            desc  = metadata_lookup.get(field, {}).get("DESCRIPTION", "")
            lines.append(
                f"  [{field}]  source={src}  logic={logic}"
                + (f"  | {desc}" if desc else "")
            )

        body = f"""
PAYER SUMMARY: {payer}  |  DATASET: {dataset}
VH Layout IDs: {', '.join(str(v) for v in vhids)}
Total mapped fields: {len(grp)}

FIELD MAPPINGS:
{chr(10).join(lines)}
""".strip()

        for i, chunk in enumerate(splitter.split_text(body)):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "sheet": "synthetic",
                    "type": "payer_summary",
                    "payer": payer.lower(),
                    "dataset_type": dataset.lower(),
                    "chunk_index": i,
                    "doc_id": stable_id(payer, dataset, "summary", str(i)),
                },
            ))

    return docs


def build_cross_payer_field_docs(df_map: pd.DataFrame) -> list[Document]:
    """
    For each CDF field, create a document listing ALL payers that map to it
    and their respective logic.  Enables queries like
    "which payers use member_id and how?".
    """
    docs = []
    splitter = build_splitter(chunk_size=700, overlap=120)

    for field, grp in df_map.groupby("CDF_FIELD"):
        lines = []
        for _, row in grp.iterrows():
            payer   = row.get("PAYER", "")
            dataset = row.get("DATASET_TYPE", "")
            source  = row.get("SOURCE", "")
            logic   = row.get("BUSINESS_LOGIC", "")
            lines.append(f"  {payer} ({dataset}): source={source}  logic={logic}")

        body = f"""
CROSS-PAYER FIELD USAGE — CDF FIELD: {field}

Payers that map to '{field}' ({len(grp)} total):
{chr(10).join(lines)}
""".strip()

        for i, chunk in enumerate(splitter.split_text(body)):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "sheet": "synthetic",
                    "type": "cross_payer_field",
                    "cdf_field": field.lower(),
                    "chunk_index": i,
                    "doc_id": stable_id(field, "cross_payer", str(i)),
                },
            ))

    return docs


# ============================================================
# MAIN PIPELINE
# ============================================================

def ingest():
    # ── Reset vectorstore ─────────────────────────────────────
    if os.path.exists(VECTORSTORE_DIR):
        shutil.rmtree(VECTORSTORE_DIR)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    # ── Load workbook ─────────────────────────────────────────
    xls = pd.ExcelFile(EXCEL_PATH)
    sheet_names = xls.sheet_names
    print(f"Sheets detected: {sheet_names}")

    all_docs: list[Document] = []
    metadata_lookup: dict = {}

    # ── Pass 1: Metadata sheet (must run first to enrich mappings) ──
    METADATA_COLS = {"CDF_FIELD", "DATA_TYPE", "FORMAT", "DESCRIPTION"}
    MAPPING_COLS  = {"DATASET_TYPE", "PAYER", "VHLAYOUTID", "SOURCE", "CDF_FIELD", "BUSINESS_LOGIC"}

    raw_sheets: dict[str, pd.DataFrame] = {}
    for sheet in sheet_names:
        df = clean_df(pd.read_excel(EXCEL_PATH, sheet_name=sheet))
        raw_sheets[sheet] = df
        cols = set(df.columns)

        if METADATA_COLS.issubset(cols) or "CDF_FIELD" in cols and "DESCRIPTION" in cols:
            print(f"  [{sheet}] → METADATA sheet")
            meta_docs, metadata_lookup = process_metadata_sheet(df, sheet)
            all_docs.extend(meta_docs)

    print(f"  Metadata lookup built: {len(metadata_lookup)} fields")

    # ── Pass 2: All other sheets ──────────────────────────────
    df_map = None
    for sheet, df in raw_sheets.items():
        cols = set(df.columns)

        if MAPPING_COLS.issubset(cols):
            print(f"  [{sheet}] → MAPPING sheet ({len(df)} rows)")
            df_map = df
            all_docs.extend(process_mapping_sheet(df, sheet, metadata_lookup))

        elif "CDF_FIELD" in cols and "DESCRIPTION" in cols:
            pass  # already handled in pass 1

        elif {"ELIG", "CLAIMS", "RXCLAIMS"}.issubset(cols) or \
             any(c in cols for c in ["ELIG", "CLAIMS", "RXCLAIMS"]):
            print(f"  [{sheet}] → CDF FIELDS sheet")
            all_docs.extend(process_cdf_fields_sheet(df, sheet))

        else:
            print(f"  [{sheet}] → GENERIC sheet (fallback)")
            splitter = build_splitter()
            for _, row in df.iterrows():
                content = "\n".join(f"{c}: {row[c]}" for c in df.columns if str(row[c]).strip())
                if not content.strip():
                    continue
                for chunk in splitter.split_text(content):
                    all_docs.append(Document(
                        page_content=f"GENERIC RECORD:\n{chunk}",
                        metadata={"sheet": sheet, "type": "generic"},
                    ))

    # ── Pass 3: Synthetic / enrichment documents ─────────────
    if df_map is not None:
        print("  Building payer summary docs …")
        all_docs.extend(build_payer_summary_docs(df_map, metadata_lookup))

        print("  Building cross-payer field docs …")
        all_docs.extend(build_cross_payer_field_docs(df_map))

    # ── Deduplicate by doc_id ─────────────────────────────────
    seen: set[str] = set()
    unique_docs: list[Document] = []
    for doc in all_docs:
        did = doc.metadata.get("doc_id", "")
        if did and did in seen:
            continue
        if did:
            seen.add(did)
        unique_docs.append(doc)

    print(f"\nTotal documents to embed: {len(unique_docs)}")

    # ── Embed + store ─────────────────────────────────────────
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    db = FAISS.from_documents(
        documents=unique_docs,
        embedding=embeddings,
    )
    db.save_local(VECTORSTORE_DIR)

    # ── Summary ───────────────────────────────────────────────
    type_counts: dict[str, int] = {}
    for doc in unique_docs:
        t = doc.metadata.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    print("\n✅ Vector DB created successfully")
    print(f"   Total documents : {len(unique_docs)}")
    print("   Breakdown by type:")
    for t, n in sorted(type_counts.items()):
        print(f"     {t:<25} {n}")


if __name__ == "__main__":
    ingest()