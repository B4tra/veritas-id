# =============================================================
# Modul: daily_scheduler.py
# Deskripsi: Script penjadwalan harian untuk menjalankan pipeline runtut
#            otomatis VERITAS-ID:
#            1. Web scraping artikel TurnBackHoax (HOAX) & Tempo CekFakta (FAKTA)
#            2. Preprocessing teks & normalisasi stopwords
#            3. Binary labelling & Train/Test split
#            4. Pelatihan ulang (retraining) & evaluasi model NLP, LSTM, CrossChecker
#            5. Pengiriman laporan email harian via Gmail SMTP
# Bagian dari: Penjadwalan Pipeline Otomatis VERITAS-ID
# =============================================================

import os
import sys
import json
import time

sys.path.append(os.path.dirname(__file__))

# Import modul-modul internal VERITAS-ID
from modules.scraper_engine import run_scraping_pipeline
from modules.preprocessor import run_preprocessing_pipeline
from modules.labelling_engine import run_labelling_pipeline
from modules.nlp_engine import HoaxDetectorModel
from modules.lstm_engine import LSTMDetectionModel
from modules.cross_checker import CrossChecker
from modules.email_service import send_gmail_report

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "data", "config.json")
LOG_PATH = os.path.join(os.path.dirname(__file__), "data", "daily_scheduler.log")

def log_message(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {msg}"
    print(log_entry)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

def run_daily_job():
    log_message("=== Memulai Eksekusi Pipeline Runtut Harian VERITAS-ID ===")
    
    # ----- Tahap 1: Scraping Mentah TurnBackHoax & Tempo CekFakta -----
    try:
        log_message("Tahap 1: Menjalankan scraping TurnBackHoax & Tempo CekFakta...")
        scrape_res = run_scraping_pipeline(max_pages=60, max_workers=10)
        log_message(f"Tahap 1 Selesai. Total Scraped: {scrape_res['total_scraped']}, FAKTA: {scrape_res['total_fakta']}, HOAX: {scrape_res['total_hoax']}")
    except Exception as e:
        log_message(f"ERROR saat Tahap 1 Scraping: {e}")
        return

    # ----- Tahap 2: Preprocessing Teks Bahasa Indonesia -----
    try:
        log_message("Tahap 2: Menjalankan preprocessing teks (Case Folding, Noise Cleaning, Stopwords Removal)...")
        prep_res = run_preprocessing_pipeline()
        log_message(f"Tahap 2 Selesai. Total Diproses: {prep_res.get('total_processed', 0)}, Rata-rata Reduksi: {prep_res.get('avg_token_reduction_pct', 0)}%")
    except Exception as e:
        log_message(f"ERROR saat Tahap 2 Preprocessing: {e}")

    # ----- Tahap 3: Labelling & Train/Test Split (80:20) -----
    try:
        log_message("Tahap 3: Menjalankan binary labelling & pembagian data train/test...")
        label_res = run_labelling_pipeline()
        log_message(f"Tahap 3 Selesai. Train Samples: {label_res.get('train_samples', 0)}, Test Samples: {label_res.get('test_samples', 0)}")
    except Exception as e:
        log_message(f"ERROR saat Tahap 3 Labelling: {e}")

    # ----- Tahap 4: Pelatihan & Evaluasi Model NLP, LSTM, Cross-Checker -----
    try:
        log_message("Tahap 4: Melatih ulang dan mengevaluasi model NLP, LSTM, dan CrossChecker...")
        nlp_model = HoaxDetectorModel()
        nlp_model.train_model()
        
        lstm_model = LSTMDetectionModel()
        lstm_model.train_model()
        
        checker = CrossChecker()
        checker.load_and_build_index()
        log_message("Tahap 4 Selesai. Seluruh model berhasil diperbarui dengan performa terbaru!")
    except Exception as e:
        log_message(f"ERROR saat Tahap 4 Retraining Model: {e}")

    # ----- Tahap 5: Kirim laporan email harian via Gmail -----
    sender = os.environ.get("SENDER_EMAIL", "").strip()
    app_pass = os.environ.get("APP_PASSWORD", "").strip()
    recipient = os.environ.get("RECIPIENT_EMAIL", "").strip()

    if not (sender and app_pass and recipient) and os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                sender = sender or config.get("sender_email", "").strip()
                app_pass = app_pass or config.get("app_password", "").strip()
                recipient = recipient or config.get("recipient_email", "").strip()
        except Exception as e:
            log_message(f"Error reading config.json: {e}")

    if sender and app_pass and recipient:
        try:
            log_message(f"Mengirimkan laporan email ke {recipient}...")
            email_res = send_gmail_report(sender, app_pass, recipient, scrape_res)
            log_message(f"Status Pengiriman Email: {email_res['message']}")
        except Exception as e:
            log_message(f"⚠️ Peringatan pengiriman email: {e}")
    else:
        log_message("ℹ️ Notifikasi email dilewati (kredensial email belum dikonfigurasi).")
        
    log_message("=== Eksekusi Penjadwalan Harian Selesai ===")

if __name__ == "__main__":
    run_daily_job()
