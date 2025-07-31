import sqlite3

def init_db():
    conn = sqlite3.connect('records.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        result TEXT,
        date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

def save_result(name, result):
    conn = sqlite3.connect('records.db')
    c = conn.cursor()
    c.execute("INSERT INTO results (name, result) VALUES (?, ?)", (name, result))
    conn.commit()
    conn.close()
