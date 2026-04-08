import pandas as pd
import sqlite3
import os

os.makedirs("db", exist_ok=True)

df = pd.read_excel("data/payers.xlsx")

conn = sqlite3.connect("db/payers.db")
df.to_sql("payers", conn, if_exists="replace", index=False)
conn.close()

print("✅ SQL DB created")