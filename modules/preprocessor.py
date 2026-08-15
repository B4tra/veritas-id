# ==============================================================================
# Modul: preprocessor.py
# Deskripsi: Modul pemrosesan teks NLP Bahasa Indonesia untuk dataset berita & hoax:
#            1. Case Folding (Konversi ke huruf kecil)
#            2. Pembersihan Noise (Tag HTML, URL, Mention, Hashtag, Karakter Spesial)
#            3. Normalisasi Simbol & Entitas (Mata uang Rp, Simbol Persen, Angka)
#            4. Penghapusan Stopwords Bahasa Indonesia (Daftar Stopwords Komprehensif)
#            5. Tokenisasi Kata & Perhitungan Statistik Pembersihan (Before vs After)
#            Dataset hasil pembersihan disimpan ke data/processed/preprocessed_dataset.csv
# Bagian dari: Tahap 2 - Preprocessing Dataset VERITAS-ID
# ==============================================================================

import os
import re
import pandas as pd
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "raw_scraped_dataset.csv")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
PROCESSED_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "preprocessed_dataset.csv")
DB_PATH = os.path.join(BASE_DIR, "data", "fact_check.db")

INDONESIAN_STOPWORDS = {
    "yang", "di", "dan", "itu", "dengan", "untuk", "tidak", "ini", "dari", "dalam",
    "akan", "pada", "juga", "saya", "ke", "karena", "tersebut", "bisa", "ada", "mereka",
    "lebih", "oleh", "saat", "sudah", "hanya", "atau", "secara", "telah", "bagi", "ia",
    "kami", "namun", "kita", "kamu", "bukan", "antara", "seperti", "jika", "sehingga",
    "dapat", "adalah", "suatu", "tentang", "maka", "ia", "lalu", "setelah", "sampai",
    "ketika", "terhadap", "dia", "anda", "apabila", "bahwa", "saja", "sebagai", "masih",
    "belum", "para", "agar", "yaitu", "yakni", "setiap", "selalu", "kembali", "sangat",
    "tanpa", "menjadi", "banyak", "lain", "sebuah", "beberapa", "harus", "selama", "ikut",
    "pun", "begitu", "pula", "lagi", "atas", "bawah", "depan", "belakang", "sekitar",
    "hal", "diri", "sendiri", "mana", "siapa", "mengapa", "bagaimana", "kapan", "dimana"
}

def case_fold(text):
    if not text:
        return ""
    return str(text).lower()

def clean_noise(text):
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+|bit\.ly/\S+', ' ', text)
    text = re.sub(r'[@#]\w+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'rp\.?\s*', ' rupiah ', text, flags=re.IGNORECASE)
    text = re.sub(r'%', ' persen ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(tokens, custom_stopwords=None):
    stopwords_set = custom_stopwords if custom_stopwords is not None else INDONESIAN_STOPWORDS
    return [t for t in tokens if t not in stopwords_set and len(t) > 1]

def tokenize_text(text):
    if not text:
        return []
    return [word for word in text.split() if word]

def preprocess_text_single(text, remove_stop=True):
    raw_str = str(text) if text else ""
    raw_token_count = len(raw_str.split())
    
    cf_text = case_fold(raw_str)
    cleaned_noise = clean_noise(cf_text)
    tokens_raw = tokenize_text(cleaned_noise)
    
    if remove_stop:
        final_tokens = remove_stopwords(tokens_raw)
    else:
        final_tokens = tokens_raw

    cleaned_text = " ".join(final_tokens)
    final_token_count = len(final_tokens)
    
    reduction_pct = 0.0
    if raw_token_count > 0:
        reduction_pct = round(((raw_token_count - final_token_count) / raw_token_count) * 100, 1)

    return {
        "raw_text": raw_str,
        "cleaned_text": cleaned_text,
        "tokens": final_tokens,
        "token_count_before": raw_token_count,
        "token_count_after": final_token_count,
        "reduction_percentage": reduction_pct
    }

def run_preprocessing_pipeline(input_csv=RAW_CSV_PATH, output_csv=PROCESSED_CSV_PATH, remove_stop=True):
    """
    Menjalankan pipeline preprocessing data secara batch untuk seluruh baris dataset mentah.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    df_raw = None
    if os.path.exists(input_csv):
        try:
            df_raw = pd.read_csv(input_csv)
        except Exception:
            pass

    if df_raw is None or df_raw.empty:
        if os.path.exists(DB_PATH):
            conn = sqlite3.connect(DB_PATH)
            df_raw = pd.read_sql_query("SELECT id, title, claim, content as raw_content, label, category, source_name, source_url, published_at FROM fact_checks", conn)
            conn.close()
        else:
            df_raw = pd.DataFrame()

    if df_raw.empty:
        return {
            "status": "warning",
            "message": "Dataset mentah kosong.",
            "total_processed": 0,
            "processed_csv_path": output_csv
        }

    processed_rows = []
    all_clean_tokens = []

    for idx, row in df_raw.iterrows():
        title_val = str(row.get("title", ""))
        claim_val = str(row.get("claim", ""))
        content_val = str(row.get("raw_content", row.get("content", "")))
        pub_date = str(row.get("published_at", "2026-06-01 10:00:00"))
        
        full_raw_text = f"{title_val} {claim_val} {content_val}".strip()
        res = preprocess_text_single(full_raw_text, remove_stop=remove_stop)
        
        processed_rows.append({
            "id": row.get("id", idx + 1),
            "source_platform": row.get("source_platform", row.get("source_name", "Unknown")),
            "original_title": title_val,
            "raw_text": full_raw_text,
            "cleaned_text": res["cleaned_text"],
            "tokens": ",".join(res["tokens"]),
            "token_count_before": res["token_count_before"],
            "token_count_after": res["token_count_after"],
            "reduction_percentage": res["reduction_percentage"],
            "label": row.get("label", "HOAX"),
            "category": row.get("category", "Umum"),
            "source_url": row.get("source_url", ""),
            "published_at": pub_date
        })
        
        if len(all_clean_tokens) < 10000:
            all_clean_tokens.extend(res["tokens"][:5])

    df_processed = pd.DataFrame(processed_rows)
    df_processed.to_csv(output_csv, index=False, encoding="utf-8")

    total_items = len(df_processed)
    avg_reduction = round(df_processed["reduction_percentage"].mean(), 2) if total_items > 0 else 0
    token_freq = pd.Series(all_clean_tokens).value_counts().head(10).to_dict()

    return {
        "status": "success",
        "total_processed": total_items,
        "avg_token_reduction_pct": avg_reduction,
        "processed_csv_path": output_csv,
        "top_frequent_words": token_freq,
        "sample_before_after": {
            "before": processed_rows[0]["raw_text"][:120] + "..." if processed_rows else "",
            "after": processed_rows[0]["cleaned_text"][:120] + "..." if processed_rows else "",
            "tokens_sample": processed_rows[0]["tokens"].split(",")[:8] if processed_rows else []
        }
    }

if __name__ == "__main__":
    res = run_preprocessing_pipeline()
    print("Preprocessing Result:", res)
