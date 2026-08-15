# ==============================================================================
# Modul: labelling_engine.py
# Deskripsi: Modul pelabelan dataset (Labelling) & Stratified Train/Test Split (80:20):
#            1. Skema Binary Labelling: 'HOAX' (1) dan 'FAKTA' (0)
#            2. Validasi & Penyelarasan Label Berdasarkan Sumber & Vonis
#            3. Pembagian Dataset Latih (80%) dan Uji (20%) secara Terstratifikasi
#            4. Penyimpanan Artefak ke data/labeled/ (labeled, train, test .csv)
# Bagian dari: Tahap 3 - Labelling Dataset VERITAS-ID
# ==============================================================================

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "preprocessed_dataset.csv")
LABELED_DATA_DIR = os.path.join(BASE_DIR, "data", "labeled")
LABELED_CSV_PATH = os.path.join(LABELED_DATA_DIR, "labeled_dataset.csv")
TRAIN_CSV_PATH = os.path.join(LABELED_DATA_DIR, "train_dataset.csv")
TEST_CSV_PATH = os.path.join(LABELED_DATA_DIR, "test_dataset.csv")
DB_PATH = os.path.join(BASE_DIR, "data", "fact_check.db")

def assign_binary_label(raw_label_str, source_str=""):
    """
    Menetapkan label biner:
    - 'HOAX' -> 1
    - 'FAKTA' -> 0
    """
    lbl = str(raw_label_str).upper()
    src = str(source_str).lower()

    if "HOAX" in lbl or "SALAH" in lbl or "PENIPUAN" in lbl or "turnbackhoax" in src:
        return "HOAX", 1
    elif "FAKTA" in lbl or "BENAR" in lbl or "KLARIFIKASI" in lbl or "tempo" in src:
        return "FAKTA", 0
    else:
        # Default fallback: jika ada indikasi hoax
        return "HOAX", 1

def run_labelling_pipeline(input_csv=PROCESSED_CSV_PATH, test_size=0.20, random_state=42):
    """
    Menjalankan pipeline pelabelan dan pemisahan data:
    1. Membaca dataset hasil preprocessing
    2. Menetapkan kolom label biner (label_text & label_num)
    3. Membagi dataset menjadi Train (80%) dan Test (20%)
    4. Menyimpan file labeled_dataset.csv, train_dataset.csv, dan test_dataset.csv
    """
    os.makedirs(LABELED_DATA_DIR, exist_ok=True)

    df = None
    if os.path.exists(input_csv):
        try:
            df = pd.read_csv(input_csv)
        except Exception as e:
            print(f"[Labelling] Gagal membaca file preprocessed: {e}")

    # Fallback: Ambil dari basis data SQLite jika preprocessed CSV kosong
    if df is None or len(df) == 0:
        if os.path.exists(DB_PATH):
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query("SELECT id, title as original_title, claim, content as raw_text, label, category, source_name as source_platform FROM fact_checks", conn)
            conn.close()
            if not df.empty and "cleaned_text" not in df.columns:
                from modules.preprocessor import preprocess_text_single
                df["cleaned_text"] = df["raw_text"].apply(lambda t: preprocess_text_single(t)["cleaned_text"])
                df["tokens"] = df["raw_text"].apply(lambda t: ",".join(preprocess_text_single(t)["tokens"]))
        else:
            df = pd.DataFrame()

    if df.empty:
        return {
            "status": "warning",
            "message": "Dataset kosong, tidak dapat melakukan labelling.",
            "total_samples": 0
        }

    # Tetapkan Binary Label
    labels_text = []
    labels_num = []

    for _, row in df.iterrows():
        lbl_str = row.get("label", "HOAX")
        src_str = str(row.get("source_platform", row.get("source_name", "")))
        lt, ln = assign_binary_label(lbl_str, src_str)
        labels_text.append(lt)
        labels_num.append(ln)

    df["label_text"] = labels_text
    df["label_num"] = labels_num

    # Simpan dataset berlabel lengkap
    df.to_csv(LABELED_CSV_PATH, index=False, encoding="utf-8")

    # Stratified Train/Test Split (80:20)
    total_len = len(df)
    stratify_target = df["label_num"] if (len(df["label_num"].unique()) > 1 and df["label_num"].value_counts().min() >= 2) else None

    if total_len >= 4:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_target
        )
    else:
        train_df = df.copy()
        test_df = df.copy()

    train_df.to_csv(TRAIN_CSV_PATH, index=False, encoding="utf-8")
    test_df.to_csv(TEST_CSV_PATH, index=False, encoding="utf-8")

    # Statistik Distribusi Label
    hoax_count = int((df["label_num"] == 1).sum())
    fakta_count = int((df["label_num"] == 0).sum())
    hoax_pct = round((hoax_count / max(total_len, 1)) * 100, 1)
    fakta_pct = round((fakta_count / max(total_len, 1)) * 100, 1)

    return {
        "status": "success",
        "total_samples": total_len,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_ratio_pct": round((len(train_df) / max(total_len, 1)) * 100, 1),
        "test_ratio_pct": round((len(test_df) / max(total_len, 1)) * 100, 1),
        "class_distribution": {
            "HOAX (1)": {"count": hoax_count, "percentage": hoax_pct},
            "FAKTA (0)": {"count": fakta_count, "percentage": fakta_pct}
        },
        "labeled_csv_path": LABELED_CSV_PATH,
        "train_csv_path": TRAIN_CSV_PATH,
        "test_csv_path": TEST_CSV_PATH
    }

def get_label_distribution():
    """Mengambil ringkasan distribusi label dari dataset berlabel."""
    if os.path.exists(LABELED_CSV_PATH):
        try:
            df = pd.read_csv(LABELED_CSV_PATH)
            total = len(df)
            hoax_c = int((df["label_num"] == 1).sum())
            fakta_c = int((df["label_num"] == 0).sum())
            return {
                "total": total,
                "hoax_count": hoax_c,
                "fakta_count": fakta_c,
                "hoax_pct": round((hoax_c / max(total, 1)) * 100, 1),
                "fakta_pct": round((fakta_c / max(total, 1)) * 100, 1)
            }
        except Exception as e:
            print(f"[Labelling Stats Error]: {e}")
    return {"total": 0, "hoax_count": 0, "fakta_count": 0, "hoax_pct": 0.0, "fakta_pct": 0.0}

if __name__ == "__main__":
    res = run_labelling_pipeline()
    print("Labelling Pipeline Result:", json.dumps(res, indent=2))
