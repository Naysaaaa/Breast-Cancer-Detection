import sqlite3

def init_db():
    conn = sqlite3.connect('records.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        result TEXT,
        confidence REAL,
        date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

def save_result(name, result, confidence=None):
    conn = sqlite3.connect('records.db')
    c = conn.cursor()
    c.execute(
        "INSERT INTO results (name, result, confidence) VALUES (?, ?, ?)",
        (name, result, confidence)
    )
    conn.commit()
    conn.close()
