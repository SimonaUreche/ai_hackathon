import sqlite3
import json

DB_PATH = "data/cvs_metadata.sqlite"

conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()

print("Tables:", cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall())

rows = cur.execute("SELECT id, filename, skills_json, industry_json FROM cvs LIMIT 5;").fetchall()
for id, fname, skills_j, ind_j in rows:
    skills = json.loads(skills_j)
    industry = json.loads(ind_j)
    print(f"{id}: {fname}")
    print("  Skills:", skills)
    print("  Industry scores:", industry)
    print()

conn.close()
