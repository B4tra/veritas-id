# 🛡️ VERITAS-ID: Intelligent Hoax Detection & Fact-Checking System

**VERITAS-ID** adalah sistem cerdas deteksi disinformasi, analisis klaim berita, dan verifikasi fakta berbasis Natural Language Processing (NLP), Deep Learning (LSTM), dan Ensemble Machine Learning yang dikembangkan khusus untuk konten berita berbahasa Indonesia.

---

## 🚀 Fitur Utama

- **🔍 Analisis Klaim & Berita Multi-Model**: Deteksi probabilitas hoax menggunakan Ensemble Classifier (TF-IDF Cosine Similarity + Bidirectional LSTM + LLM Reasoning).
- **🕷️ Multi-Source Scraper Otomatis**: Pengumpulan artikel cek fakta harian dari sumber terpercaya (TurnBackHoax Mafindo, Tempo Cek Fakta, Kompas Cek Fakta, dll.).
- **⚙️ Pipeline Preprocessing Lengkap**: Case folding, pembersihan noise karakter/URL, normalisasi simbol/mata uang, filtering stopwords, dan tokenisasi kata.
- **🏷️ Binary Labelling & Stratified Splitting**: Pelabelan biner otomatis (*HOAX* = 1, *FAKTA* = 0) dan pembagian dataset terstratifikasi (80% Train, 20% Test).
- **🧠 Deep Learning & NLP Engine**:
  - Model LSTM dengan Word Embeddings & Dropout Regularization.
  - TF-IDF Vectorizer + Cosine Similarity cross-checking terhadap basis data cek fakta terverifikasi.
- **📊 Interactive Streamlit Dashboard**: Antarmuka modern untuk eksplorasi dataset, visualisasi tren misinformasi, evaluasi metrik model, dan verifikasi instan.
- **📧 Notifikasi Otomatis (Email Alert)**: Pengiriman ringkasan temuan hoax berkala ke email pengguna.

---

## 📁 Struktur Direktori

```text
veritas-id/
├── app.py                      # Aplikasi antarmuka utama Streamlit Dashboard
├── daily_scheduler.py          # Scheduler otomatis crawler berita & pembaruan database
├── requirements.txt            # Dependensi Python
├── README.md                   # Dokumentasi proyek
├── assets/
│   └── style.css               # Styling UI Streamlit kustom
├── data/
│   ├── fact_check.db           # Basis data SQLite utama untuk arsip cek fakta
│   ├── config.json.example     # Contoh konfigurasi API dan SMTP
│   ├── raw/
│   │   └── raw_scraped_dataset.csv       # Dataset mentah hasil scraping
│   ├── processed/
│   │   └── preprocessed_dataset.csv     # Dataset hasil pra-pemrosesan NLP
│   └── labeled/
│       ├── labeled_dataset.csv          # Dataset lengkap berlabel biner
│       ├── train_dataset.csv            # Dataset pelatihan (80%)
│       └── test_dataset.csv             # Dataset pengujian (20%)
└── modules/
    ├── preprocessor.py         # Pipeline pra-pemrosesan NLP teks Bahasa Indonesia
    ├── labelling_engine.py     # Engine pelabelan data biner & stratified split
    ├── scraper_engine.py       # Engine web scraping artikel cek fakta
    ├── nlp_engine.py           # Ekstraksi fitur teks, TF-IDF, & similarity matching
    ├── lstm_engine.py          # Arsitektur & pelatihan model neural network LSTM
    ├── ensemble_engine.py      # Penggabungan prediksi multi-model (Ensemble)
    ├── cross_checker.py        # Verifikasi silang klaim dengan basis data lokal
    ├── llm_extractor.py        # Integrasi LLM untuk analisis semantik mendalam
    ├── email_service.py        # Modul pengiriman notifikasi alert via SMTP
    ├── database.py             # Operasi CRUD SQLite database
    ├── seed_data.py            # Inisialisasi data awal fact-check
    └── url_parser.py           # Utilitas ekstraksi konten dari URL berita
```

---

## 🛠️ Panduan Instalasi & Penggunaan

### 1. Kloning Repositori
```bash
git clone https://github.com/B4tra/veritas-id.git
cd veritas-id
```

### 2. Buat Lingkungan Virtual (Virtual Environment)
Disarankan menggunakan Python 3.10 atau 3.11:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Pasang Dependensi
```bash
pip install -r requirements.txt
```

### 4. Konfigurasi (Opsional)
Salin berkas konfigurasi contoh jika ingin mengaktifkan fitur notifikasi email dan API LLM:
```bash
cp data/config.json.example data/config.json
```
Edit `data/config.json` dan isi kredensial yang sesuai.

### 5. Jalankan Aplikasi Web
```bash
streamlit run app.py
```
Aplikasi akan terbuka otomatis di peramban Anda pada alamat `http://localhost:8501`.

---

## 🔬 Menjalankan Pipeline secara Mandiri

- **Scraping Data**:
  ```bash
  python modules/scraper_engine.py
  ```
- **Preprocessing NLP**:
  ```bash
  python modules/preprocessor.py
  ```
- **Labelling & Train/Test Split**:
  ```bash
  python modules/labelling_engine.py
  ```
- **Pelatihan Model LSTM**:
  ```bash
  python modules/lstm_engine.py
  ```

---

## 👥 Kontribusi & Lisensi
Dikembangkan oleh Tim VERITAS-ID untuk riset dan edukasi literasi digital di Indonesia.
