# =============================================================
# Modul: daily_scheduler.py
# Deskripsi: Script penjadwalan harian untuk menjalankan pipeline
#            otomatis VERITAS-ID yang meliputi:
#            1. Web scraping artikel berita terbaru (HOAX & FAKTA)
#            2. Pelatihan ulang (retraining) model NLP & LSTM
#            3. Pengiriman laporan email harian via Gmail SMTP
#            Script ini dirancang untuk dijalankan oleh
#            Windows Task Scheduler setiap hari secara otomatis.
# Bagian dari: Proyek VERITAS-ID - Sistem Deteksi Hoax Indonesia
# =============================================================

import os
import sys
import json
import time

# Menambahkan path direktori utama agar modul-modul di folder modules/ bisa di-import
sys.path.append(os.path.dirname(__file__))

# Import modul-modul internal VERITAS-ID
from modules.scraper_engine import run_scraping_pipeline
from modules.nlp_engine import HoaxDetectorModel
from modules.lstm_engine import LSTMDetectionModel
from modules.cross_checker import CrossChecker
from modules.email_service import send_gmail_report

# Path ke file konfigurasi (berisi kredensial email, API key, dll)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "data", "config.json")
# Path ke file log untuk mencatat aktivitas penjadwalan harian
LOG_PATH = os.path.join(os.path.dirname(__file__), "data", "daily_scheduler.log")

# Fungsi untuk mencatat pesan log ke console dan file log
def log_message(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {msg}"
    print(log_entry)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

# Fungsi utama yang menjalankan seluruh pipeline harian VERITAS-ID
# Terdiri dari 3 langkah: scraping → retraining model → kirim email laporan
def run_daily_job():
    log_message("=== Memulai Eksekusi Penjadwalan Harian VERITAS-ID ===")
    
    # ----- Langkah 1: Jalankan pipeline web scraping -----
    # Mengambil artikel terbaru dari TurnBackHoax (HOAX), CNN Indonesia (FAKTA),
    # dan ANTARA News (FAKTA), lalu menyimpannya ke database SQLite
    try:
        scrape_res = run_scraping_pipeline(limit_per_source=5)
        log_message(f"Scraping selesai. Total terambil: {scrape_res['total_scraped']}, Artikel Baru Dimasukkan: {scrape_res['inserted_count']}")
    except Exception as e:
        log_message(f"ERROR saat menjalankan scraping pipeline: {e}")
        return

    # ----- Langkah 2: Latih ulang semua model deteksi -----
    # Model NLP (Logistic Regression), LSTM (Deep Learning), dan
    # CrossChecker (TF-IDF index) diperbarui dengan data terbaru dari database
    try:
        log_message("Melatih ulang model NLP & LSTM...")
        nlp_model = HoaxDetectorModel()
        nlp_model.train_model()
        
        lstm_model = LSTMDetectionModel()
        lstm_model.train_model()
        
        # Membangun ulang index TF-IDF untuk cross-check klaim
        checker = CrossChecker()
        checker.load_and_build_index()
        log_message("Model NLP, LSTM, dan CrossChecker berhasil diperbarui!")
    except Exception as e:
        log_message(f"ERROR saat retraining model: {e}")

    # ----- Langkah 3: Kirim laporan email harian via Gmail -----
    # Kredensial email diambil dari environment variable terlebih dahulu,
    # jika tidak ditemukan maka fallback ke file config.json
    sender = os.environ.get("SENDER_EMAIL", "").strip()
    app_pass = os.environ.get("APP_PASSWORD", "").strip()
    recipient = os.environ.get("RECIPIENT_EMAIL", "").strip()

    # Fallback: baca kredensial dari config.json jika environment variable kosong
    if not (sender and app_pass and recipient) and os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                sender = sender or config.get("sender_email", "").strip()
                app_pass = app_pass or config.get("app_password", "").strip()
                recipient = recipient or config.get("recipient_email", "").strip()
        except Exception as e:
            log_message(f"Error reading config.json: {e}")

    # Kirim email laporan jika semua kredensial tersedia
    if sender and app_pass and recipient:
        try:
            log_message(f"Mengirimkan laporan email ke {recipient}...")
            email_res = send_gmail_report(sender, app_pass, recipient, scrape_res)
            log_message(f"Status Pengiriman Email: {email_res['message']}")
            if not email_res.get("success"):
                log_message(f"❌ GAGAL MENGIRIM EMAIL: {email_res.get('message')}")
                sys.exit(1)
        except Exception as e:
            log_message(f"❌ ERROR saat mengirim email: {e}")
            sys.exit(1)
    else:
        # Hentikan skrip jika kredensial email tidak lengkap
        log_message("❌ Kredensial email (SENDER_EMAIL, APP_PASSWORD, RECIPIENT_EMAIL) belum ditemukan atau belum lengkap! Menghentikan skrip.")
        sys.exit(1)
        
    log_message("=== Penjadwalan Harian Selesai ===")

# Entry point: langsung jalankan pipeline harian saat script dieksekusi
if __name__ == "__main__":
    run_daily_job()
