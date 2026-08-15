# ==============================================================================
# Modul: seed_data.py
# Deskripsi: Modul data awal (seed corpus) yang berisi sampel artikel terverifikasi
#            secara berimbang antara TurnBackHoax.id (HOAX) dan CekFakta Tempo (FAKTA).
#            Digunakan saat inisialisasi awal database dan pipeline dataset.
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
    # ==================== KELAS HOAX (TurnBackHoax.id) ====================
    {
        "title": "[HOAX] Makan Telur Rebus Jam 12 Malam Menyembuhkan Corona",
        "claim": "Memakan telur rebus pada jam 12 malam secara ajaib dapat menangkal dan menyembuhkan infeksi virus Corona",
        "content": "Beredar pesan berantai di WhatsApp yang mengklaim bahwa memakan telur rebus pada jam 12 malam secara ajaib dapat menangkal dan menyembuhkan infeksi virus Corona. Klaim ini tidak memiliki dasar ilmiah dan telah dibantah oleh WHO serta Kementerian Kesehatan RI.",
        "label": "HOAX",
        "category": "Kesehatan",
        "source_name": "TurnBackHoax.id (MAFINDO)",
        "source_url": "https://turnbackhoax.id/articles/telur-rebus-corona",
        "verdict_details": "SALAH. Tidak ada bukti ilmiah bahwa telur rebus dapat menyembuhkan COVID-19. WHO menegaskan tidak ada makanan yang bisa menyembuhkan Corona.",
        "source_platform": "TurnBackHoax.id"
    },
    {
        "title": "[HOAX] Vaksin COVID-19 Mengandung Microchip Bill Gates",
        "claim": "Vaksin COVID-19 mengandung microchip yang ditanamkan Bill Gates untuk mengontrol manusia",
        "content": "Beredar klaim di media sosial bahwa vaksin COVID-19 mengandung microchip buatan Bill Gates Foundation untuk mengendalikan umat manusia. Klaim ini adalah hoaks yang telah dibantah oleh banyak lembaga kesehatan dunia termasuk WHO dan CDC.",
        "label": "HOAX",
        "category": "Kesehatan",
        "source_name": "TurnBackHoax.id (MAFINDO)",
        "source_url": "https://turnbackhoax.id/articles/vaksin-microchip",
        "verdict_details": "SALAH. Vaksin COVID-19 tidak mengandung microchip. Bill Gates Foundation mendukung pengembangan vaksin untuk kesehatan global.",
        "source_platform": "TurnBackHoax.id"
    },
    {
        "title": "[HOAX] Minum Air Hangat Campur Lemon Membunuh Virus Corona",
        "claim": "Minum air hangat campur lemon setiap pagi dapat membunuh virus Corona dalam tubuh",
        "content": "Pesan berantai WhatsApp menyebarkan klaim bahwa minum air hangat dicampur perasan lemon setiap pagi bisa membunuh virus Corona. Dokter dan ahli kesehatan menegaskan ini tidak benar. Air lemon baik untuk kesehatan tapi tidak membunuh virus.",
        "label": "HOAX",
        "category": "Kesehatan",
        "source_name": "TurnBackHoax.id (MAFINDO)",
        "source_url": "https://turnbackhoax.id/articles/lemon-bunuh-virus",
        "verdict_details": "SALAH. Air lemon tidak memiliki kemampuan membunuh virus Corona. WHO menyatakan tidak ada minuman yang bisa menyembuhkan COVID-19.",
        "source_platform": "TurnBackHoax.id"
    },
    {
        "title": "[HOAX] 5G Menyebabkan Penyebaran Virus Corona",
        "claim": "Jaringan 5G menyebabkan penyebaran virus Corona dan melemahkan sistem imun tubuh manusia",
        "content": "Viral di media sosial teori konspirasi yang mengklaim jaringan 5G menjadi penyebab penyebaran virus Corona dan melemahkan imunitas tubuh. Klaim ini sudah dibantah oleh para ilmuwan dan WHO. Virus menyebar melalui droplet bukan gelombang radio.",
        "label": "HOAX",
        "category": "Teknologi & Bencana",
        "source_name": "TurnBackHoax.id (MAFINDO)",
        "source_url": "https://turnbackhoax.id/articles/5g-corona-konspirasi",
        "verdict_details": "SALAH. Tidak ada hubungan antara jaringan 5G dan penyebaran virus Corona. Virus menyebar melalui droplet pernapasan.",
        "source_platform": "TurnBackHoax.id"
    },
    {
        "title": "[PENIPUAN] Link Pendaftaran Bantuan Pemerintah Rp 600 Ribu",
        "claim": "Pemerintah memberikan bantuan tunai Rp 600 ribu melalui link pendaftaran online di WhatsApp",
        "content": "Beredar tautan mencurigakan di WhatsApp yang mengaku sebagai link pendaftaran bantuan pemerintah sebesar Rp 600 ribu. Segeralah daftar sebelum kuota habis. Tautan ini adalah penipuan phishing yang mencuri data pribadi korban.",
        "label": "HOAX",
        "category": "Penipuan Online",
        "source_name": "TurnBackHoax.id (MAFINDO)",
        "source_url": "https://turnbackhoax.id/articles/bansos-palsu-wa",
        "verdict_details": "PENIPUAN. Pemerintah tidak pernah mendistribusikan bantuan sosial melalui tautan WhatsApp. Ini adalah modus phishing pencurian data.",
        "source_platform": "TurnBackHoax.id"
    },
    {
        "title": "[PENIPUAN] Lowongan Kerja Palsu E-Commerce Gaji 15 Juta",
        "claim": "Platform e-commerce membuka lowongan kerja online dari rumah dengan gaji Rp 15 juta per bulan tanpa pengalaman",
        "content": "Viral di media sosial iklan lowongan kerja online dengan tawaran gaji Rp 15 juta per bulan dari rumah tanpa pengalaman. Segera daftar melalui link bit.ly. Pihak manajemen mengonfirmasi ini adalah penipuan dan bukan lowongan resmi.",
        "label": "HOAX",
        "category": "Penipuan Online",
        "source_name": "TurnBackHoax.id (MAFINDO)",
        "source_url": "https://turnbackhoax.id/articles/loker-palsu-15juta",
        "verdict_details": "PENIPUAN. Tidak pernah ada penawaran lowongan kerja instan berbayar melalui tautan bit.ly. Ini modus penipuan online.",
        "source_platform": "TurnBackHoax.id"
    },
    {
        "title": "[HOAX] Sinar Kosmik Berbahaya Memasuki Bumi Malam Ini",
        "claim": "NASA mengumumkan sinar kosmik berbahaya akan memasuki bumi malam ini dan semua orang harus mematikan HP",
        "content": "Pesan berantai WhatsApp menyebarkan peringatan palsu bahwa NASA mengumumkan sinar kosmik berbahaya akan memasuki atmosfer bumi malam ini. Segeralah matikan HP dan jauhi perangkat elektronik. Sebarkanlah pesan ini ke seluruh keluarga. NASA membantah pernah mengeluarkan peringatan semacam itu.",
        "label": "HOAX",
        "category": "Bencana Alam",
        "source_name": "TurnBackHoax.id (MAFINDO)",
        "source_url": "https://turnbackhoax.id/articles/sinar-kosmik-nasa-palsu",
        "verdict_details": "SALAH. NASA tidak pernah mengeluarkan peringatan sinar kosmik berbahaya. Pesan ini adalah hoaks berulang yang beredar sejak bertahun-tahun.",
        "source_platform": "TurnBackHoax.id"
    },
    {
        "title": "[HOAX] Gempa Besar 9.5 SR Akan Menghantam Jawa Dalam 3 Hari",
        "claim": "BMKG memprediksi gempa besar 9.5 SR akan menghantam Pulau Jawa dalam 3 hari ke depan",
        "content": "Viral pesan yang mengatasnamakan BMKG memperingatkan gempa besar 9.5 SR akan menghantam Jawa dalam 3 hari. Segeralah evakuasi ke tempat tinggi. BMKG menegaskan gempa tidak bisa diprediksi dan tidak pernah mengeluarkan peringatan semacam itu.",
        "label": "HOAX",
        "category": "Bencana Alam",
        "source_name": "TurnBackHoax.id (MAFINDO)",
        "source_url": "https://turnbackhoax.id/articles/gempa-prediksi-bmkg-hoax",
        "verdict_details": "SALAH. BMKG menegaskan gempa bumi tidak dapat diprediksi waktu dan lokasinya secara presisi. Pesan ini adalah hoaks.",
        "source_platform": "TurnBackHoax.id"
    },
    {
        "title": "[HOAX] Mi Instan Mengandung Lilin yang Menyebabkan Kanker",
        "claim": "Mi instan dilapisi lilin yang tidak bisa dicerna tubuh dan menyebabkan kanker usus",
        "content": "Pesan berantai di media sosial mengklaim bahwa mi instan dilapisi lilin untuk mencegahnya menempel. Lilin ini diklaim tidak bisa dicerna oleh tubuh selama 3 hari dan menyebabkan kanker usus. BPOM dan ahli gizi membantah klaim ini karena mi instan menggunakan minyak goreng bukan lilin.",
        "label": "HOAX",
        "category": "Kesehatan",
        "source_name": "TurnBackHoax.id (MAFINDO)",
        "source_url": "https://turnbackhoax.id/articles/mi-instan-lilin-kanker",
        "verdict_details": "SALAH. Mi instan tidak mengandung lapisan lilin. Lapisan berminyak berasal dari proses penggorengan higienis berstandar BPOM.",
        "source_platform": "TurnBackHoax.id"
    },
    {
        "title": "[HOAX] Dokumen Rahasia Bocor Terkait Vaksinasi",
        "claim": "Beredar bocoran dokumen resmi bahwa vaksinasi menyebabkan kelumpuhan organ",
        "content": "Sebuah dokumen pdf diedit sedemikian rupa dan diklaim sebagai laporan investigasi kementerian terkait efek samping vaksin. Lembaga resmi memastikan dokumen tersebut adalah hasil rekayasa dokumen palsu.",
        "label": "HOAX",
        "category": "Kesehatan & Regulasi",
        "source_name": "TurnBackHoax.id (MAFINDO)",
        "source_url": "https://turnbackhoax.id/articles/dokumen-rekayasa-vaksin",
        "verdict_details": "SALAH / FABRIKASI. Dokumen yang disebarkan tidak memiliki nomor registrasi sah dan merupakan manipulasi digital.",
        "source_platform": "TurnBackHoax.id"
    },

    # ==================== KELAS FAKTA (Tempo CekFakta) ====================
    {
        "title": "[Tempo] Mengapa Kerugian Korupsi Timah Menciut Tinggal Rp 29 Triliun",
        "claim": "Kejaksaan Agung menghitung ulang kerugian negara pada skandal tata niaga timah berdasarkan audit BPKP",
        "content": "Kejaksaan Agung bersama Badan Pengawasan Keuangan dan Pembangunan (BPKP) merilis audit komprehensif mengenai estimasi kerugian keuangan negara serta kerusakan lingkungan pada perkara dugaan korupsi tata niaga komoditas timah. Proses hukum kini berjalan di Pengadilan Tindak Pidana Korupsi.",
        "label": "FAKTA",
        "category": "Hukum & Korupsi",
        "source_name": "Tempo CekFakta",
        "source_url": "https://cekfakta.tempo.co/hukum/mengapa-kerugian-korupsi-timah-menciut",
        "verdict_details": "TERVERIFIKASI FAKTA. Fakta persidangan dan pernyataan resmi jaksa penuntut umum Kejaksaan Agung RI.",
        "source_platform": "CekFakta Tempo"
    },
    {
        "title": "[Tempo] Klarifikasi Regulasi Bisnis WNA di Indonesia Berdasarkan UU PMA",
        "claim": "Pemerintah menegaskan kepemilikan bisnis oleh warga negara asing harus memenuhi syarat penanaman modal asing (PMA)",
        "content": "Kementerian Investasi / BKPM menegaskan bahwa seluruh warga negara asing yang hendak mendirikan dan menjalankan usaha di wilayah Republik Indonesia wajib menaati ketentuan izin tinggal serta batasan modal minimum PT PMA sesuai ketentuan hukum ketenagakerjaan dan perizinan terpadu OSS.",
        "label": "FAKTA",
        "category": "Regulasi & Ekonomi",
        "source_name": "Tempo CekFakta",
        "source_url": "https://cekfakta.tempo.co/ekonomi/klarifikasi-bisnis-wna-pma",
        "verdict_details": "TERVERIFIKASI FAKTA. Sesuai dengan Undang-Undang Penanaman Modal dan petunjuk teknis Kementerian Investasi/BKPM.",
        "source_platform": "CekFakta Tempo"
    },
    {
        "title": "[Tempo] Inflasi Indonesia Terjaga pada Rentang Sasaran Bank Indonesia",
        "claim": "Bank Indonesia dan BPS mencatat inflasi indeks harga konsumen berada dalam sasaran target 2.5 plus minus 1 persen",
        "content": "Berdasarkan rilis resmi Badan Pusat Statistik (BPS) dan Bank Indonesia, tingkat inflasi IHK nasional terkendali berkat sinergi Tim Pengendalian Inflasi Pusat dan Daerah (TPIP dan TPID) melalui program Gerakan Nasional Pengendalian Inflasi Pangan di berbagai provinsi.",
        "label": "FAKTA",
        "category": "Ekonomi & Keuangan",
        "source_name": "Tempo CekFakta",
        "source_url": "https://cekfakta.tempo.co/ekonomi/inflasi-terkendali-bi-bps",
        "verdict_details": "TERVERIFIKASI FAKTA. Data resmi dirilis berkala oleh BPS dan Bank Indonesia dalam konferensi pers bulanan.",
        "source_platform": "CekFakta Tempo"
    },
    {
        "title": "[Tempo] Pemerintah Tegaskan Sanksi Pidana Bagi Pelaku Penyelenggara Judi Online",
        "claim": "Satgas Pemberantasan Perjudian Daring memblokir ribuan rekening bank dan situs yang terafiliasi judi online",
        "content": "Kementerian Komunikasi dan Digital bersama Satgas Judi Online dan Otoritas Jasa Keuangan (OJK) terus melakukan pemblokiran rekening bank mencurigakan serta take down terhadap ribuan konten promosi judi online di ruang digital nasional.",
        "label": "FAKTA",
        "category": "Hukum & Digital",
        "source_name": "Tempo CekFakta",
        "source_url": "https://cekfakta.tempo.co/digital/satgas-judi-online-blokir-rekening",
        "verdict_details": "TERVERIFIKASI FAKTA. Konfirmasi resmi dari Menkomdigi, OJK, dan Bareskrim Polri dalam konferensi pers bersama.",
        "source_platform": "CekFakta Tempo"
    },
    {
        "title": "[Tempo] BMKG Jelaskan Dinamika Atmosfer dan Potensi Cuaca Ekstrem Musim Pancaroba",
        "claim": "BMKG mempublikasikan analisis peringatan dini hujan lebat dan angin kencang di beberapa wilayah Indonesia",
        "content": "Badan Meteorologi, Klimatologi, dan Geofisika (BMKG) merilis prospek cuaca mingguan yang menunjukkan potensi peningkatan curah hujan akibat fenomena gelombang atmosfer Kelvin dan Rossby Equatorial di wilayah ekuator Indonesia.",
        "label": "FAKTA",
        "category": "Sains & Lingkungan",
        "source_name": "Tempo CekFakta",
        "source_url": "https://cekfakta.tempo.co/sains/bmkg-prospek-cuaca-ekstrem",
        "verdict_details": "TERVERIFIKASI FAKTA. Berdasarkan siaran pers resmi kedeputian meteorologi BMKG melalui kanal resmi bmkg.go.id.",
        "source_platform": "CekFakta Tempo"
    },
    {
        "title": "[Tempo] Kejaksaan Agung Tetapkan Tersangka Kasus Dugaan Suap dan Gratifikasi",
        "claim": "Penyidik Jampidsus Kejagung menahan oknum yang diduga menerima gratifikasi dalam penanganan perkara peradilan",
        "content": "Penyidik Jaksa Agung Muda Bidang Tindak Pidana Khusus (Jampidsus) resmi menetapkan tersangka baru dan melakukan penahanan 20 hari ke depan untuk kepentingan penyidikan dugaan korupsi dan gratifikasi terkait peradilan.",
        "label": "FAKTA",
        "category": "Hukum & Korupsi",
        "source_name": "Tempo CekFakta",
        "source_url": "https://cekfakta.tempo.co/hukum/kejagung-tetapkan-tersangka-gratifikasi",
        "verdict_details": "TERVERIFIKASI FAKTA. Disampaikan langsung oleh Kepala Pusat Penerangan Hukum (Kapuspenkum) Kejaksaan Agung RI.",
        "source_platform": "CekFakta Tempo"
    },
    {
        "title": "[Tempo] Menkes Pastikan Ketersediaan Vaksin Imunisasi Dasar Lengkap Nasional",
        "claim": "Kementerian Kesehatan menjamin distribusi vaksin imunisasi rutin anak tetap terpenuhi di seluruh Puskesmas",
        "content": "Menteri Kesehatan Republik Indonesia memastikan pasokan vaksin imunisasi dasar lengkap bagi bayi dan anak tersedia secara memadai di fasilitas kesehatan tingkat pertama (FKTP) dan puskesmas di seluruh provinsi.",
        "label": "FAKTA",
        "category": "Kesehatan",
        "source_name": "Tempo CekFakta",
        "source_url": "https://cekfakta.tempo.co/kesehatan/kemenkes-vaksin-imunisasi-lengkap",
        "verdict_details": "TERVERIFIKASI FAKTA. Kemenkes RI menyampaikan klarifikasi dan data distribusi logistik vaksin nasional.",
        "source_platform": "CekFakta Tempo"
    },
    {
        "title": "[Tempo] Perusahaan Jasa Angkut Sampah Liar Dikenai Sanksi Denda Tegas",
        "claim": "Dinas Lingkungan Hidup menjatuhkan denda administratif terhadap pengelola pembuangan sampah ilegal",
        "content": "Pemerintah Daerah melalui Dinas Lingkungan Hidup melakukan penindakan tegas berupa denda administratif dan penyegelan lokasi terhadap oknum pembuang sampah liar yang melanggar Peraturan Daerah tentang Pengelolaan Sampah.",
        "label": "FAKTA",
        "category": "Lingkungan",
        "source_name": "Tempo CekFakta",
        "source_url": "https://cekfakta.tempo.co/lingkungan/penindakan-tps-liar-denda",
        "verdict_details": "TERVERIFIKASI FAKTA. Penegakan hukum perda yang terdokumentasi resmi oleh Satpol PP dan DLH.",
        "source_platform": "CekFakta Tempo"
    }
]

def init_database(db_path=DB_PATH):
    """Inisialisasi tabel database SQLite dan isi data awal jika belum ada."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Tabel 1: fact_checks (Dataset Cek Fakta Utama)
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

    # Check published_at
    try:
        cursor.execute("ALTER TABLE fact_checks ADD COLUMN published_at TEXT")
    except sqlite3.OperationalError:
        pass

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

    # Periksa apakah data awal perlu diisi
    cursor.execute("SELECT COUNT(*) FROM fact_checks")
    total_count = cursor.fetchone()[0]
    if total_count < 10:
        insert_seed_data(db_path, conn)

    conn.close()

def insert_seed_data(db_path=DB_PATH, conn=None):
    """Memasukkan data seed TurnBackHoax dan Tempo ke dalam database SQLite."""
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
            cursor.execute("""
                INSERT INTO fact_checks (title, claim, content, label, category, source_name, source_url, verdict_details, published_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item["title"], item["claim"], item["content"], item["label"],
                item["category"], item["source_name"], item["source_url"], item["verdict_details"],
                now_str, now_str
            ))
            inserted += 1

    conn.commit()
    if close_conn:
        conn.close()

    print(f"[VERITAS-ID Seed] Berhasil memuat {inserted} data awal TurnBackHoax & Tempo.")
    return inserted

if __name__ == "__main__":
    init_database(DB_PATH)
