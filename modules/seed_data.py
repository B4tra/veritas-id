import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fact_check.db")

SEED_DATA = []

def init_database(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Table 1: Fact check dataset
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fact_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            claim TEXT NOT NULL,
            content TEXT NOT NULL,
            label TEXT NOT NULL,
            category TEXT,
            source_name TEXT,
            source_url TEXT,
            verdict_details TEXT,
            published_at TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Check if published_at column exists in existing DBs
    try:
        cursor.execute("ALTER TABLE fact_checks ADD COLUMN published_at TEXT")
    except sqlite3.OperationalError:
        pass # Column already exists
        
    # Table 2: User search history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT NOT NULL,
            input_type TEXT NOT NULL, -- 'text' or 'url'
            predicted_label TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            matched_reference_id INTEGER,
            checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (matched_reference_id) REFERENCES fact_checks (id)
        )
    """)
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    db_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fact_check.db")
    init_database(db_file)
