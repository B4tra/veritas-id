# Modul ini merupakan mesin klasifikasi teks berbasis NLP untuk mendeteksi hoax.
# Pendekatan yang digunakan adalah ekstraksi fitur teks dengan TF-IDF dipadukan
# dengan model prediktif Logistic Regression. Modul ini juga memanfaatkan fitur
# heuristik sebagai sinyal penguat deteksi hoax.
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from modules.database import fetch_all_fact_checks

# Daftar kata kunci yang sering muncul pada pesan hoax sebagai sinyal deteksi
# Common Indonesian stopwords & sensational hoax signal keywords
HOAX_SIGNALS = [
    "segeralah", "sebarkanlah", "viral", "awas", "bahaya", "ajaib", "meratakan",
    "malam ini", "jam 12", "gratis", "hadiah", "klaim", "100 juta", "rahasia",
    "dituduh", "babi", "kanker", "mematikan", "segera tukarkan", "porcine",
    "sinar kosmik", "evakuasi", "bit.ly", "whatsapp", "pesan berantai",
    # Kata tambahan untuk deteksi hoax
    "penipuan", "waspada", "darurat", "terbukti", "dijamin", "terungkap",
    "mengejutkan", "mencengangkan", "geger", "heboh", "gempar", "konspirasi",
    "tersembunyi", "dirahasiakan", "microchip", "racun", "berbahaya",
    "sebarkan", "viralkan", "tolong sebarkan", "kirimkan ke", "forward",
    "lilin", "plastik", "palsu", "bohong", "hoax", "hoaks",
    "daftar segera", "kuota terbatas", "jangan sampai", "buruan",
    "obat ajaib", "sembuh total", "tanpa efek samping", "herbal alami",
    "prediksi gempa", "bubar", "terpecah", "manipulasi", "photoshop"
]

# Fungsi untuk membersihkan dan menormalisasi teks masukan
def preprocess_text(text):
    """Clean and normalize Indonesian input text while preserving key entities and numbers."""
    if not text:
        return ""
    text = text.lower()
    # Mengganti tautan URL dengan penanda khusus
    # Replace URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' url_link ', text)
    # Menormalisasi simbol mata uang rupiah dan persentase menjadi teks
    # Normalize currency & percentage symbols to words
    text = re.sub(r'rp\.?\s*', ' rp ', text)
    text = re.sub(r'%', ' persen ', text)
    # Menghapus karakter khusus kecuali huruf, angka, dan spasi
    # Remove special non-alphanumeric characters except basic spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Menghilangkan spasi ganda agar teks lebih rapi
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Fungsi untuk mengekstrak fitur tambahan seperti tanda baca dan huruf kapital
def extract_text_features(raw_text):
    """Extract additional heuristic features from raw text for hoax detection."""
    if not raw_text:
        return 0.0, 0.0, 0.0, 0
    
    # Menghitung jumlah tanda seru sebagai indikator gaya bahasa sensasional
    # Count exclamation marks (sensationalism indicator)
    exclamation_count = raw_text.count('!')
    
    # Menghitung rasio huruf kapital yang biasa digunakan untuk memberi penekanan berlebih
    # Calculate CAPS ratio (shouting indicator)
    alpha_chars = [c for c in raw_text if c.isalpha()]
    caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)
    
    # Mencari dan menghitung kata-kata yang merupakan sinyal hoax
    # Count hoax signal words found
    text_lower = raw_text.lower()
    signal_count = sum(1 for sig in HOAX_SIGNALS if sig in text_lower)
    
    # Mendeteksi keberadaan tautan atau URL dalam teks (indikator phishing)
    # URL/link presence (phishing indicator)
    url_count = len(re.findall(r'https?://\S+|www\.\S+|bit\.ly', raw_text, re.IGNORECASE))
    
    return exclamation_count, caps_ratio, signal_count, url_count

# Kelas utama yang membungkus logika pelatihan dan prediksi menggunakan Logistic Regression
class HoaxDetectorModel:
    def __init__(self):
        # Inisialisasi vectorizer TF-IDF dan model regresi logistik
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        # class_weight='balanced' gives more weight to minority class (HOAX)
        self.model = LogisticRegression(C=1.5, random_state=42, class_weight='balanced')
        self.is_trained = False
        self.train_model()

    # Fungsi untuk melatih model berdasarkan data fakta dan hoax dari database
    def train_model(self):
        """Train the classifier model using the fact check seed corpus."""
        records = fetch_all_fact_checks()
        if not records:
            return

        texts = []
        labels = []
        # Menyiapkan data latih dengan menggabungkan judul, klaim, dan konten
        for r in records:
            clean_t = preprocess_text(f"{r['title']} {r['claim']} {r['content']}")
            texts.append(clean_t)
            labels.append(r['label']) # 'HOAX' or 'FAKTA'

        # Memastikan terdapat minimal dua kelas label (HOAX dan FAKTA) sebelum melatih model
        if len(set(labels)) < 2:
            return

        # Melakukan transformasi teks ke vektor TF-IDF lalu melatih model
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.is_trained = True

    # Fungsi untuk memprediksi probabilitas hoax pada teks masukan pengguna
    def predict(self, raw_text):
        """
        Predict whether raw_text is HOAX or FAKTA.
        Returns dict containing predicted_label, confidence_score, probabilities, and suspicious_words.
        """
        clean_input = preprocess_text(raw_text)
        
        # Mengembalikan status peringatan jika teks terlalu pendek untuk dianalisis
        if len(clean_input.split()) < 3:
            return {
                "label": "TIDAK CUKUP DATA",
                "badge_type": "warning",
                "confidence_score": 0.0,
                "hoax_prob": 0.0,
                "fakta_prob": 0.0,
                "suspicious_words": [],
                "note": "Input terlalu pendek untuk dianalisis oleh model NLP."
            }

        # Pastikan model sudah dilatih sebelum melakukan prediksi
        if not self.is_trained:
            self.train_model()

        # Mendapatkan probabilitas dasar dari model klasifikasi
        X_input = self.vectorizer.transform([clean_input])
        probs = self.model.predict_proba(X_input)[0]
        classes = list(self.model.classes_)

        hoax_idx = classes.index("HOAX") if "HOAX" in classes else 0
        fakta_idx = classes.index("FAKTA") if "FAKTA" in classes else 1

        hoax_prob = float(probs[hoax_idx])
        fakta_prob = float(probs[fakta_idx])

        # Mengekstrak fitur teks untuk menghitung bobot tambahan (heuristic boost)
        # Apply heuristic boost from text-level features
        excl_count, caps_ratio, signal_count, url_count = extract_text_features(raw_text)
        
        # Logika heuristic boost: Menambahkan bobot penalti pada teks yang memiliki karakteristik hoax
        # Boost hoax probability based on heuristic signals
        heuristic_boost = 0.0
        if signal_count >= 3:
            heuristic_boost += 0.10
        elif signal_count >= 1:
            heuristic_boost += 0.05
        if excl_count >= 3:
            heuristic_boost += 0.03
        if caps_ratio > 0.3:
            heuristic_boost += 0.03
        if url_count >= 1 and signal_count >= 1:
            heuristic_boost += 0.05
        
        # Menyesuaikan probabilitas akhir jika terdapat peningkatan bobot dari heuristik
        # Apply boost (shift probability towards HOAX)
        if heuristic_boost > 0:
            hoax_prob = min(hoax_prob + heuristic_boost, 0.99)
            fakta_prob = 1.0 - hoax_prob

        # Mengekstrak kata-kata yang mencurigakan beserta tingkat kepentingannya
        # Feature importance / keyword highlight extraction
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        input_vector = X_input.toarray()[0]
        nonzero_indices = np.where(input_vector > 0)[0]
        
        word_scores = []
        coefs = self.model.coef_[0] if hasattr(self.model, "coef_") else None

        # Menghitung seberapa besar pengaruh sebuah kata terhadap prediksi hoax
        for idx in nonzero_indices:
            word = feature_names[idx]
            tfidf_val = input_vector[idx]
            
            # Memberikan bobot tambahan apabila kata tersebut terdaftar dalam sinyal hoax
            # Boost score if word is in known sensational hoax signals list
            signal_boost = 2.5 if any(sig in word for sig in HOAX_SIGNALS) else 1.0
            
            coef_val = coefs[idx] if coefs is not None else 0.0
            # Mengalikan TF-IDF dengan koefisien regresi logistik sebagai skor akhir
            # Positive coef -> HOAX tendency
            importance = coef_val * tfidf_val * signal_boost
            word_scores.append((word, importance))

        # Mengurutkan kata berdasarkan skor kepentingannya lalu mengambil maksimal 8 kata
        # Sort words by hoax impact
        word_scores.sort(key=lambda x: x[1], reverse=True)
        suspicious_words = [w for w, score in word_scores if score > 0.01 or any(sig in w for sig in HOAX_SIGNALS)][:8]

        # Menentukan label dan tipe lencana peringatan berdasarkan batas probabilitas 50%
        # Determine final classification label (threshold 0.50 for balanced)
        if hoax_prob >= 0.50:
            label = "BERPOTENSI HOAX"
            badge_type = "danger"
            confidence = round(hoax_prob * 100, 1)
        else:
            label = "KEMUNGKINAN FAKTA"
            badge_type = "success"
            confidence = round(fakta_prob * 100, 1)

        return {
            "label": label,
            "badge_type": badge_type,
            "confidence_score": confidence,
            "hoax_prob": round(hoax_prob * 100, 1),
            "fakta_prob": round(fakta_prob * 100, 1),
            "suspicious_words": list(set(suspicious_words)),
            "heuristic_signals": {
                "exclamation_count": excl_count,
                "caps_ratio": round(caps_ratio, 3),
                "hoax_signal_words": signal_count,
                "url_count": url_count,
                "boost_applied": round(heuristic_boost, 3)
            },
            "note": "Model NLP dilatih menggunakan korpus data terverifikasi TurnBackHoax, Tempo, dan CNN Indonesia (class_weight=balanced)."
        }
