import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fact_check.db")

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def fetch_all_fact_checks():
    """Retrieve all fact check entries from SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM fact_checks ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def save_search_history(input_text, input_type, predicted_label, confidence_score, matched_reference_id=None):
    """Save a user query to search_history table."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO search_history (input_text, input_type, predicted_label, confidence_score, matched_reference_id)
        VALUES (?, ?, ?, ?, ?)
    """, (input_text, input_type, predicted_label, float(confidence_score), matched_reference_id))
    conn.commit()
    conn.close()

def get_recent_history(limit=10):
    """Fetch recent search history queries."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.*, f.title as ref_title, f.source_name as ref_source
        FROM search_history s
        LEFT JOIN fact_checks f ON s.matched_reference_id = f.id
        ORDER BY s.checked_at DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
