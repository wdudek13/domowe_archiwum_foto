import sqlite3

conn = sqlite3.connect("data/db/media_vault.db")
cursor = conn.cursor()
cursor.execute("PRAGMA integrity_check;")
print(cursor.fetchall())