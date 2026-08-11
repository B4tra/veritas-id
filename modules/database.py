# Modul ini menangani koneksi dan query ke database SQLite.
# Digunakan untuk menyimpan data cek fakta dan riwayat pencarian pengguna.
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fact_check.db")

# Membuat atau mengambil koneksi ke database SQLite
def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Mengambil semua data cek fakta dari database beserta konfigurasinya
def fetch_all_fact_checks(order_by="created_at DESC", *args, **kwargs):
    """Retrieve all fact check entries from SQLite with specified time/id ordering."""
    if not order_by and "order_by" in kwargs:
        order_by = kwargs["order_by"]
    if not isinstance(order_by, str):
        order_by = "created_at DESC"
        
    conn = get_connection()
    cursor = conn.cursor()
    # Validate order_by to prevent SQL injection
    allowed_orders = {
        "created_at DESC": "created_at DESC, id DESC",
        "created_at ASC": "created_at ASC, id ASC",
        "published_at DESC": "published_at DESC, id DESC",
        "published_at ASC": "published_at ASC, id ASC",
        "id ASC": "id ASC",
        "id DESC": "id DESC"
    }
    # Menentukan urutan query SQL untuk menghindari injeksi SQL
    sql_order = allowed_orders.get(order_by, "created_at DESC, id DESC")
    cursor.execute(f"SELECT * FROM fact_checks ORDER BY {sql_order}")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# Menyimpan riwayat pencarian dari pengguna (teks/input dan hasil prediksi)
def save_search_history(input_text, input_type, predicted_label, confidence_score, matched_reference_id=None):
    """Save a user query to search_history table."""
    conn = get_connection()
    cursor = conn.cursor()
    # Melakukan insert data baru ke tabel search_history
    cursor.execute("""
        INSERT INTO search_history (input_text, input_type, predicted_label, confidence_score, matched_reference_id)
        VALUES (?, ?, ?, ?, ?)
    """, (input_text, input_type, predicted_label, float(confidence_score), matched_reference_id))
    conn.commit()
    conn.close()

# Mendapatkan sejumlah riwayat pencarian terbaru beserta referensinya
def get_recent_history(limit=10):
    """Fetch recent search history queries."""
    conn = get_connection()
    cursor = conn.cursor()
    # Menggabungkan data dari tabel search_history dengan fact_checks
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
