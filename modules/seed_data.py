import sqlite3
import json
import os
import time

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fact_check.db")

SEED_DATA = [
    # --- HOAX KESEHATAN ---
    {
        "title": "[HOAX] Makan Telur Rebus Jam 12 Malam Menyembuhkan Corona",
        "claim": "Memakan telur rebus pada jam 12 malam secara ajaib dapat menangkal dan menyembuhkan infeksi virus Corona",
        "content": "Beredar pesan berantai di WhatsApp yang mengklaim bahwa memakan telur rebus pada jam 12 malam secara ajaib dapat menangkal dan menyembuhkan infeksi virus Corona. Klaim ini tidak memiliki dasar ilmiah dan telah dibantah oleh WHO serta Kementerian Kesehatan RI.",
        "label": "HOAX",
        "category": "Kesehatan",
        "source_name": "TurnBackHoax.id (MAFINDO)",
        "source_url": "https://turnbackhoax.id/",
        "verdict_details": "SALAH. Tidak ada bukti ilmiah bahwa telur rebus dapat menyembuhkan COVID-19. WHO menegaskan tidak ada makanan yang bisa menyembuhkan Corona."
    },
    {
        "title": "[HOAX] Vaksin COVID-19 Mengandung Microchip Bill Gates",
        "claim": "Vaksin COVID-19 mengandung microchip yang ditanamkan Bill Gates untuk mengontrol manusia",
        "content": "Beredar klaim di media sosial bahwa vaksin COVID-19 mengandung microchip buatan Bill Gates Foundation untuk mengendalikan umat manusia. Klaim ini adalah hoaks yang telah dibantah oleh banyak lembaga kesehatan dunia termasuk WHO dan CDC.",
        "label": "HOAX",
        "category": "Kesehatan",
        "source_name": "TurnBackHoax.id (MAFINDO)",
        "source_url": "https://turnbackhoax.id/",
        "verdict_details": "SALAH. Vaksin COVID-19 tidak mengandung microchip. Bill Gates Foundation mendukung pengembangan vaksin untuk kesehatan global."
    },
    {
        "title": "[HOAX] Minum Air Hangat Campur Lemon Membunuh Virus Corona",
        "claim": "Minum air hangat campur lemon setiap pagi dapat membunuh virus Corona dalam tubuh",
        "content": "Pesan berantai WhatsApp menyebarkan klaim bahwa minum air hangat dicampur perasan lemon setiap pagi bisa membunuh virus Corona. Dokter dan ahli kesehatan menegaskan ini tidak benar. Air lemon baik untuk kesehatan tapi tidak membunuh virus.",
        "label": "HOAX",
        "category": "Kesehatan",
        "source_name": "Tempo CekFakta",
        "source_url": "https://cekfakta.tempo.co/",
        "verdict_details": "SALAH. Air lemon tidak memiliki kemampuan membunuh virus Corona. WHO menyatakan tidak ada minuman yang bisa menyembuhkan COVID-19."
    },
    {
        "title": "[HOAX] 5G Menyebabkan Penyebaran Virus Corona",
        "claim": "Jaringan 5G menyebabkan penyebaran virus Corona dan melemahkan sistem imun tubuh manusia",
        "content": "Viral di media sosial teori konspirasi yang mengklaim jaringan 5G menjadi penyebab penyebaran virus Corona dan melemahkan imunitas tubuh. Klaim ini sudah dibantah oleh para ilmuwan dan WHO. Virus menyebar melalui droplet bukan gelombang radio.",
        "label": "HOAX",
        "category": "Kesehatan",
        "source_name": "TurnBackHoax.id (MAFINDO)",
        "source_url": "https://turnbackhoax.id/",
        "verdict_details": "SALAH. Tidak ada hubungan antara jaringan 5G dan penyebaran virus Corona. Virus menyebar melalui droplet pernapasan."
    },
    # --- HOAX PENIPUAN ---
    {
        "title": "[PENIPUAN] Link Pendaftaran Bantuan Pemerintah Rp 600 Ribu",
        "claim": "Pemerintah memberikan bantuan tunai Rp 600 ribu melalui link pendaftaran online di WhatsApp",
        "content": "Beredar tautan mencurigakan di WhatsApp yang mengaku sebagai link pendaftaran bantuan pemerintah sebesar Rp 600 ribu. Segeralah daftar sebelum kuota habis. Tautan ini adalah penipuan phishing yang mencuri data pribadi korban.",
        "label": "HOAX",
        "category": "Penipuan Online",
        "source_name": "Kominfo RI",
        "source_url": "https://turnbackhoax.id/",
        "verdict_details": "PENIPUAN. Pemerintah tidak pernah mendistribusikan bantuan sosial melalui tautan WhatsApp. Ini adalah modus phishing pencurian data."
    },
    {
        "title": "[PENIPUAN] Lowongan Kerja Palsu Shopee Gaji 15 Juta",
        "claim": "Shopee membuka lowongan kerja online dari rumah dengan gaji Rp 15 juta per bulan tanpa pengalaman",
        "content": "Viral di media sosial iklan lowongan kerja yang mengatasnamakan Shopee dengan tawaran gaji Rp 15 juta per bulan, bisa kerja dari rumah tanpa pengalaman. Segera daftar melalui link bit.ly. Shopee mengonfirmasi ini adalah penipuan dan bukan lowongan resmi.",
        "label": "HOAX",
        "category": "Penipuan Online",
        "source_name": "TurnBackHoax.id (MAFINDO)",
        "source_url": "https://turnbackhoax.id/",
        "verdict_details": "PENIPUAN. Shopee tidak pernah menawarkan lowongan kerja melalui media sosial atau tautan bit.ly. Ini modus penipuan."
    },
    {
        "title": "[PENIPUAN] Undian Berhadiah Telkomsel Rp 100 Juta",
        "claim": "Telkomsel mengadakan undian berhadiah Rp 100 juta dan pemenang harus mengirimkan pulsa untuk klaim hadiah",
        "content": "Beredar SMS dan pesan WhatsApp yang mengaku dari Telkomsel memberitahukan bahwa nomor Anda memenangkan undian Rp 100 juta. Untuk mengklaim hadiah segera hubungi nomor berikut dan kirimkan pulsa sebagai biaya administrasi. Telkomsel menegaskan ini penipuan.",
        "label": "HOAX",
        "category": "Penipuan Online",
        "source_name": "TurnBackHoax.id (MAFINDO)",
        "source_url": "https://turnbackhoax.id/",
        "verdict_details": "PENIPUAN. Telkomsel tidak pernah meminta pelanggan mengirimkan pulsa untuk mengklaim hadiah undian."
    },
    # --- HOAX POLITIK & SOSIAL ---
    {
        "title": "[HOAX] Foto Manipulasi Capres Bertemu Tokoh Kontroversial",
        "claim": "Foto menunjukkan calon presiden bertemu dengan tokoh kontroversial di luar negeri membuktikan konspirasi politik",
        "content": "Beredar foto di media sosial yang diklaim menunjukkan salah satu capres bertemu tokoh kontroversial internasional. Setelah ditelusuri menggunakan reverse image search, foto tersebut ternyata hasil manipulasi dan editan Photoshop. Foto asli berasal dari konteks berbeda.",
        "label": "HOAX",
        "category": "Politik",
        "source_name": "Tempo CekFakta",
        "source_url": "https://cekfakta.tempo.co/",
        "verdict_details": "SALAH. Foto tersebut adalah hasil manipulasi digital. Foto asli berasal dari acara berbeda dan telah diedit."
    },
    {
        "title": "[HOAX] Indonesia Bubar Tahun 2030 Menurut CIA",
        "claim": "CIA memprediksi Indonesia akan bubar dan terpecah menjadi beberapa negara pada tahun 2030",
        "content": "Viral di media sosial klaim bahwa CIA Amerika Serikat memprediksi Indonesia akan bubar dan terpecah menjadi beberapa negara kecil pada tahun 2030. Klaim ini menggunakan dokumen palsu dan telah dibantah oleh para ahli hubungan internasional.",
        "label": "HOAX",
        "category": "Politik",
        "source_name": "TurnBackHoax.id (MAFINDO)",
        "source_url": "https://turnbackhoax.id/",
        "verdict_details": "SALAH. Tidak ada laporan resmi CIA yang memprediksi Indonesia bubar. Dokumen yang beredar adalah palsu."
    },
    {
        "title": "[HOAX] Video Antrean Sembako Sepanjang 5 Km di Jakarta",
        "claim": "Video menunjukkan antrean pembagian sembako sepanjang 5 kilometer di Jakarta akibat krisis ekonomi parah",
        "content": "Beredar video di TikTok dan WhatsApp yang mengklaim menunjukkan antrean pembagian sembako gratis sepanjang 5 km di Jakarta sebagai bukti krisis ekonomi parah. Video tersebut sebenarnya diambil dari antrean vaksinasi massal di stadion pada tahun sebelumnya.",
        "label": "HOAX",
        "category": "Politik",
        "source_name": "CNN Indonesia CekFakta",
        "source_url": "https://www.cnnindonesia.com/",
        "verdict_details": "KELIRU KONTEKS. Video bukan antrean sembako melainkan antrean vaksinasi massal di stadion."
    },
    # --- HOAX BENCANA & CUACA ---
    {
        "title": "[HOAX] Sinar Kosmik Berbahaya Memasuki Bumi Malam Ini",
        "claim": "NASA mengumumkan sinar kosmik berbahaya akan memasuki bumi malam ini dan semua orang harus mematikan HP",
        "content": "Pesan berantai WhatsApp menyebarkan peringatan palsu bahwa NASA mengumumkan sinar kosmik berbahaya akan memasuki atmosfer bumi malam ini. Segeralah matikan HP dan jauhi perangkat elektronik. Sebarkanlah pesan ini ke seluruh keluarga. NASA membantah pernah mengeluarkan peringatan semacam itu.",
        "label": "HOAX",
        "category": "Bencana Alam",
        "source_name": "TurnBackHoax.id (MAFINDO)",
        "source_url": "https://turnbackhoax.id/",
        "verdict_details": "SALAH. NASA tidak pernah mengeluarkan peringatan sinar kosmik berbahaya. Pesan ini adalah hoaks berulang yang beredar sejak bertahun-tahun."
    },
    {
        "title": "[HOAX] Gempa Besar 9.5 SR Akan Menghantam Jawa Dalam 3 Hari",
        "claim": "BMKG memprediksi gempa besar 9.5 SR akan menghantam Pulau Jawa dalam 3 hari ke depan",
        "content": "Viral pesan yang mengatasnamakan BMKG memperingatkan gempa besar 9.5 SR akan menghantam Jawa dalam 3 hari. Segeralah evakuasi ke tempat tinggi. BMKG menegaskan gempa tidak bisa diprediksi dan tidak pernah mengeluarkan peringatan semacam itu.",
        "label": "HOAX",
        "category": "Bencana Alam",
        "source_name": "BMKG",
        "source_url": "https://turnbackhoax.id/",
        "verdict_details": "SALAH. BMKG menegaskan gempa bumi tidak dapat diprediksi waktu dan lokasinya. Pesan ini hoaks."
    },
    # --- HOAX MAKANAN ---
    {
        "title": "[HOAX] Mi Instan Mengandung Lilin yang Menyebabkan Kanker",
        "claim": "Mi instan dilapisi lilin yang tidak bisa dicerna tubuh dan menyebabkan kanker usus",
        "content": "Pesan berantai di media sosial mengklaim bahwa mi instan dilapisi lilin untuk mencegahnya menempel. Lilin ini diklaim tidak bisa dicerna oleh tubuh selama 3 hari dan menyebabkan kanker usus. BPOM dan ahli gizi membantah klaim ini karena mi instan menggunakan minyak goreng bukan lilin.",
        "label": "HOAX",
        "category": "Kesehatan",
        "source_name": "BPOM RI",
        "source_url": "https://turnbackhoax.id/",
        "verdict_details": "SALAH. Mi instan tidak mengandung lilin. Lapisan berminyak pada mi instan berasal dari proses penggorengan."
    },
    {
        "title": "[HOAX] Nasi Kemarin yang Dipanaskan Ulang Lebih Berbahaya dari Racun",
        "claim": "Nasi sisa kemarin yang dipanaskan ulang mengandung racun arsenik dan lebih berbahaya dari sianida",
        "content": "Beredar klaim bahwa memakan nasi sisa kemarin yang telah dipanaskan ulang sangat berbahaya karena mengandung racun arsenik yang lebih mematikan dari sianida. Ahli gizi menjelaskan nasi kemarin aman dikonsumsi asalkan disimpan dengan benar di kulkas dan dipanaskan sempurna.",
        "label": "HOAX",
        "category": "Kesehatan",
        "source_name": "Tempo CekFakta",
        "source_url": "https://cekfakta.tempo.co/",
        "verdict_details": "KELIRU. Nasi kemarin aman dikonsumsi jika disimpan dengan benar. Klaim arsenik lebih berbahaya dari sianida tidak berdasar."
    },
    {
        "title": "[HOAX] Plastik Ditemukan dalam Nasi Bungkus Warteg",
        "claim": "Ditemukan butiran plastik dalam nasi bungkus di warteg yang dicampur untuk menekan biaya beras",
        "content": "Video viral menunjukkan seseorang menemukan butiran plastik dalam nasi bungkus warteg. Video ini diklaim sebagai bukti bahwa pedagang mencampurkan plastik dengan beras untuk menekan biaya. Setelah ditelusuri, butiran tersebut adalah pati beras yang menggumpal, bukan plastik.",
        "label": "HOAX",
        "category": "Kesehatan",
        "source_name": "TurnBackHoax.id (MAFINDO)",
        "source_url": "https://turnbackhoax.id/",
        "verdict_details": "SALAH. Butiran putih bukan plastik melainkan pati beras yang menggumpal akibat proses pemasakan."
    }
]

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
    
    # Insert seed data if table is empty or has very few HOAX entries
    cursor.execute("SELECT COUNT(*) FROM fact_checks WHERE label = 'HOAX'")
    hoax_count = cursor.fetchone()[0]
    if hoax_count < 15:
        insert_seed_data(db_path, conn)
    
    conn.close()

def insert_seed_data(db_path, conn=None):
    """Insert manual seed HOAX data into the database if not already present."""
    close_conn = False
    if conn is None:
        conn = sqlite3.connect(db_path)
        close_conn = True
    
    cursor = conn.cursor()
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    inserted = 0
    
    for item in SEED_DATA:
        cursor.execute("SELECT COUNT(*) FROM fact_checks WHERE title = ? OR claim = ?", 
                      (item["title"], item["claim"]))
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
    
    print(f"[VERITAS-ID Seed] Inserted {inserted} new HOAX seed records.")
    return inserted

if __name__ == "__main__":
    db_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fact_check.db")
    init_database(db_file)
