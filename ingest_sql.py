import pandas as pd
import sqlite3
import os

os.makedirs("db", exist_ok=True)

df = pd.read_excel("data/payers.xlsx")

# Clean dataset
df = df.fillna("").drop_duplicates()

for col in df.columns:
    df[col] = df[col].astype(str).str.strip()

conn = sqlite3.connect("db/payers.db")

df.to_sql(
    "payers",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("✅ SQL DB created")
