# ==============================================================================
# Modul: seed_data.py
# Deskripsi: Modul inisialisasi basis data SQLite dan data awal contoh (seed).
#            Hanya menyisakan 1 sampel representatif kelas HOAX (TurnBackHoax.id)
#            dan 1 sampel representatif kelas FAKTA (CekFakta Tempo) tahun 2026.
# Bagian dari: Fondasi Dataset & Database VERITAS-ID
# ==============================================================================

import sqlite3
import os
import time
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "fact_check.db")
RAW_CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "raw_scraped_dataset.csv")

SEED_DATA = [
    # ==================== 1 CONTOH SAMPEL HOAX (TurnBackHoax.id) ====================
    {
        "title": "[PENIPUAN] Tautan Pendaftaran Bantuan Sosial Tunai 2026",
        "claim": "Pemerintah membuka pendaftaran bantuan sosial tunai 2026 sebesar Rp 600.000 melalui tautan pendaftaran online WhatsApp dan Telegram",
        "content": "Beredar pesan berantai di WhatsApp yang mengklaim bahwa pemerintah membagikan bantuan sosial tunai 2026 sebesar Rp 600.000 melalui tautan online. Faktanya, Kementerian Sosial menegaskan tidak pernah membuka pendaftaran bansos via tautan tidak resmi. Ini adalah modus penipuan phishing pencurian data.",
        "label": "HOAX",
        "category": "Penipuan Online",
        "source_name": "TurnBackHoax.id (MAFINDO)",
        "source_url": "https://turnbackhoax.id/articles/bansos-palsu-2026",
        "verdict_details": "PENIPUAN / HOAX. Kementerian Sosial RI tidak pernah membuka pendaftaran bantuan sosial melalui tautan WhatsApp atau pihak ketiga.",
        "source_platform": "TurnBackHoax.id",
        "published_at": "2026-03-15 10:00:00"
    },

    # ==================== 1 CONTOH SAMPEL FAKTA (CekFakta Tempo) ====================
    {
        "title": "[Tempo] Bank Indonesia Pertahankan BI-Rate 6 Persen untuk Menjaga Stabilitas Rupiah 2026",
        "claim": "Bank Indonesia memutuskan mempertahankan BI-Rate pada level 6,00 persen untuk menjaga stabilitas nilai tukar Rupiah pada 2026",
        "content": "Rapat Dewan Gubernur Bank Indonesia memutuskan untuk mempertahankan BI-Rate sebesar 6,00 persen guna memperkuat stabilitas nilai tukar Rupiah dan memastikan inflasi tetap terkendali dalam sasaran 2,5 persen pada tahun 2026.",
        "label": "FAKTA",
        "category": "Tempo - Ekonomi & Keuangan",
        "source_name": "Tempo CekFakta",
        "source_url": "https://cekfakta.tempo.co/ekonomi/bi-rate-2026",
        "verdict_details": "TERVERIFIKASI FAKTA. Informasi resmi dirilis melalui Siaran Pers Rapat Dewan Gubernur Bank Indonesia.",
        "source_platform": "CekFakta Tempo",
        "published_at": "2026-04-20 14:00:00"
    }
]

def init_database(db_path=DB_PATH):
    """
    Menginisialisasi tabel database SQLite:
    - fact_checks (Korpus Artikel Rujukan Cek Fakta)
    - search_history (Riwayat Pengecekan Klaim Pengguna)
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Tabel 1: fact_checks (Artikel Rujukan)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fact_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            claim TEXT,
            content TEXT,
            label TEXT NOT NULL,
            category TEXT DEFAULT 'Umum',
            source_name TEXT,
            source_url TEXT,
            verdict_details TEXT,
            published_at TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Tabel 2: search_history (Riwayat Pengecekan)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT NOT NULL,
            input_type TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            matched_reference_id INTEGER,
            checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (matched_reference_id) REFERENCES fact_checks (id)
        )
    """)
    conn.commit()

    # Periksa jumlah data di basis data
    cursor.execute("SELECT COUNT(*) FROM fact_checks")
    total_count = cursor.fetchone()[0]

    # Jika database kosong, muat dari file CSV mentah (jika ada) atau masukkan 2 contoh sampel seed
    if total_count == 0:
        if os.path.exists(RAW_CSV_PATH):
            try:
                df_raw = pd.read_csv(RAW_CSV_PATH)
                for _, row in df_raw.iterrows():
                    cursor.execute("""
                        INSERT INTO fact_checks (title, claim, content, label, category, source_name, source_url, verdict_details, published_at, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row.get("title", ""), row.get("claim", ""), row.get("raw_content", row.get("content", "")),
                        row.get("label", "HOAX"), row.get("category", "Umum"), row.get("source_name", "TurnBackHoax.id"),
                        row.get("source_url", ""), row.get("verdict_details", ""),
                        row.get("published_at", "2026-06-01 10:00:00"), row.get("crawled_at", time.strftime("%Y-%m-%d %H:%M:%S"))
                    ))
                conn.commit()
                print(f"[VERITAS-ID DB] Berhasil memuat {len(df_raw)} data awal dari CSV mentah.")
            except Exception:
                insert_seed_data(db_path, conn)
        else:
            insert_seed_data(db_path, conn)

    conn.close()

def insert_seed_data(db_path=DB_PATH, conn=None):
    """Memasukkan 2 sampel contoh seed (1 HOAX & 1 FAKTA) ke dalam database SQLite."""
    close_conn = False
    if conn is None:
        conn = sqlite3.connect(db_path)
        close_conn = True

    cursor = conn.cursor()
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    inserted = 0

    for item in SEED_DATA:
        cursor.execute("SELECT COUNT(*) FROM fact_checks WHERE title = ? OR claim = ?", (item["title"], item["claim"]))
        if cursor.fetchone()[0] == 0:
            pub_date = item.get("published_at", now_str)
            cursor.execute("""
                INSERT INTO fact_checks (title, claim, content, label, category, source_name, source_url, verdict_details, published_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item["title"], item["claim"], item["content"], item["label"],
                item["category"], item["source_name"], item["source_url"], item["verdict_details"],
                pub_date, now_str
            ))
            inserted += 1

    conn.commit()
    if close_conn:
        conn.close()

    print(f"[VERITAS-ID Seed] Berhasil memuat {inserted} sampel data awal (1 HOAX & 1 FAKTA).")
    return inserted

if __name__ == "__main__":
    init_database(DB_PATH)
