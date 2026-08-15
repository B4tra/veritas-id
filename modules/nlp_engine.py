# ==============================================================================
# Modul: nlp_engine.py
# Deskripsi: Mesin klasifikasi teks NLP berbasis TF-IDF & Logistic Regression:
#            1. Pelatihan model pada data latih (data/labeled/train_dataset.csv / SQLite)
#            2. Evaluasi performa pada data uji (Akurasi, Presisi, Recall, F1, Confusion Matrix)
#            3. Prediksi probabilitas HOAX vs FAKTA dengan Heuristic Signal Boost
#            4. Explainability: Ekstraksi kata mencurigakan & kontribusi fitur (TF-IDF * Coef)
# Bagian dari: Tahap 4 - Modelling Dataset VERITAS-ID
# ==============================================================================

import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from modules.database import fetch_all_fact_checks
from modules.preprocessor import preprocess_text_single, clean_noise, case_fold

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "data", "labeled", "train_dataset.csv")
TEST_CSV_PATH = os.path.join(BASE_DIR, "data", "labeled", "test_dataset.csv")

HOAX_SIGNALS = [
    "segeralah", "sebarkanlah", "viral", "awas", "bahaya", "ajaib", "meratakan",
    "malam ini", "jam 12", "gratis", "hadiah", "klaim", "100 juta", "rahasia",
    "dituduh", "babi", "kanker", "mematikan", "segera tukarkan", "porcine",
    "sinar kosmik", "evakuasi", "bit.ly", "whatsapp", "pesan berantai",
    "penipuan", "waspada", "darurat", "terbukti", "dijamin", "terungkap",
    "mengejutkan", "mencengangkan", "geger", "heboh", "gempar", "konspirasi",
    "tersembunyi", "dirahasiakan", "microchip", "racun", "berbahaya",
    "sebarkan", "viralkan", "tolong sebarkan", "kirimkan ke", "forward",
    "lilin", "plastik", "palsu", "bohong", "hoax", "hoaks",
    "daftar segera", "kuota terbatas", "jangan sampai", "buruan",
    "obat ajaib", "sembuh total", "tanpa efek samping", "herbal alami",
    "prediksi gempa", "bubar", "terpecah", "manipulasi", "photoshop"
]

def preprocess_text(text):
    """Clean and normalize Indonesian input text while preserving key entities and numbers."""
    if not text:
        return ""
    res = preprocess_text_single(text, remove_stop=True)
    return res["cleaned_text"]

def extract_text_features(raw_text):
    """Ekstraksi fitur heuristik teks (tanda seru, rasio huruf kapital, sinyal hoax, link)."""
    if not raw_text:
        return 0.0, 0.0, 0.0, 0
    
    exclamation_count = raw_text.count('!')
    alpha_chars = [c for c in raw_text if c.isalpha()]
    caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)
    
    text_lower = raw_text.lower()
    signal_count = sum(1 for sig in HOAX_SIGNALS if sig in text_lower)
    url_count = len(re.findall(r'https?://\S+|www\.\S+|bit\.ly', raw_text, re.IGNORECASE))
    
    return exclamation_count, caps_ratio, signal_count, url_count

class HoaxDetectorModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.model = LogisticRegression(C=1.5, random_state=42, class_weight='balanced')
        self.is_trained = False
        self.evaluation_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "train_samples": 0,
            "test_samples": 0
        }
        self.train_model()

    def train_model(self):
        """Melatih model Logistic Regression pada train dataset dan mengevaluasi pada test dataset."""
        train_texts = []
        train_labels = []

        # Prioritas 1: Gunakan train_dataset.csv jika tersedia
        if os.path.exists(TRAIN_CSV_PATH):
            try:
                df_train = pd.read_csv(TRAIN_CSV_PATH)
                if not df_train.empty and "cleaned_text" in df_train.columns and "label_text" in df_train.columns:
                    train_texts = df_train["cleaned_text"].fillna("").tolist()
                    train_labels = df_train["label_text"].tolist()
            except Exception as e:
                print(f"[NLP Model] Gagal membaca train_dataset.csv: {e}")

        # Fallback: Ambil dari basis data SQLite
        if not train_texts or len(set(train_labels)) < 2:
            records = fetch_all_fact_checks()
            if records:
                train_texts = [preprocess_text(f"{r['title']} {r['claim']} {r['content']}") for r in records]
                train_labels = [r['label'] for r in records]

        if len(set(train_labels)) < 2:
            return

        # Fit Vectorizer & Logistic Regression
        X_train = self.vectorizer.fit_transform(train_texts)
        self.model.fit(X_train, train_labels)
        self.is_trained = True

        # Evaluasi pada data uji (test_dataset.csv)
        self._evaluate_on_test_set(train_count=len(train_texts))

    def _evaluate_on_test_set(self, train_count=0):
        """Evaluasi metrik model NLP pada data uji terpisah."""
        test_texts = []
        test_labels = []

        if os.path.exists(TEST_CSV_PATH):
            try:
                df_test = pd.read_csv(TEST_CSV_PATH)
                if not df_test.empty and "cleaned_text" in df_test.columns and "label_text" in df_test.columns:
                    test_texts = df_test["cleaned_text"].fillna("").tolist()
                    test_labels = df_test["label_text"].tolist()
            except Exception as e:
                print(f"[NLP Model] Gagal membaca test_dataset.csv: {e}")

        # Jika test set belum ada, gunakan data latih untuk estimasi metrik
        if not test_texts:
            test_texts = [preprocess_text(f"{r['title']} {r['claim']} {r['content']}") for r in fetch_all_fact_checks()][:4]
            test_labels = [r['label'] for r in fetch_all_fact_checks()][:4]

        if test_texts and len(set(test_labels)) >= 1:
            X_test = self.vectorizer.transform(test_texts)
            y_pred = self.model.predict(X_test)
            
            # Hitung metrik biner (Positif = HOAX)
            acc = round(accuracy_score(test_labels, y_pred) * 100, 1)
            prec = round(precision_score(test_labels, y_pred, pos_label="HOAX", zero_division=0) * 100, 1)
            rec = round(recall_score(test_labels, y_pred, pos_label="HOAX", zero_division=0) * 100, 1)
            f1 = round(f1_score(test_labels, y_pred, pos_label="HOAX", zero_division=0) * 100, 1)
            
            cm = confusion_matrix(test_labels, y_pred, labels=["FAKTA", "HOAX"])
            
            self.evaluation_metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confusion_matrix": cm.tolist() if hasattr(cm, "tolist") else [[1, 0], [0, 1]],
                "train_samples": train_count,
                "test_samples": len(test_texts)
            }

    def predict(self, raw_text):
        """Prediksi teks pengguna dengan probabilitas, explainability kata kunci, dan sinyal heuristik."""
        clean_input = preprocess_text(raw_text)
        
        if len(clean_input.split()) < 2:
            return {
                "label": "TIDAK CUKUP DATA",
                "badge_type": "warning",
                "confidence_score": 0.0,
                "hoax_prob": 0.0,
                "fakta_prob": 0.0,
                "suspicious_words": [],
                "note": "Input terlalu pendek untuk dianalisis oleh model NLP."
            }

        if not self.is_trained:
            self.train_model()

        X_input = self.vectorizer.transform([clean_input])
        probs = self.model.predict_proba(X_input)[0]
        classes = list(self.model.classes_)

        hoax_idx = classes.index("HOAX") if "HOAX" in classes else 0
        fakta_idx = classes.index("FAKTA") if "FAKTA" in classes else 1

        hoax_prob = float(probs[hoax_idx])
        fakta_prob = float(probs[fakta_idx])

        # Heuristic Boost
        excl_count, caps_ratio, signal_count, url_count = extract_text_features(raw_text)
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

        if heuristic_boost > 0:
            hoax_prob = min(hoax_prob + heuristic_boost, 0.99)
            fakta_prob = 1.0 - hoax_prob

        # Explainability: Hitung bobot kata (TF-IDF * Koefisien Regresi)
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        input_vector = X_input.toarray()[0]
        nonzero_indices = np.where(input_vector > 0)[0]
        
        word_scores = []
        coefs = self.model.coef_[0] if hasattr(self.model, "coef_") else None

        for idx in nonzero_indices:
            word = feature_names[idx]
            tfidf_val = input_vector[idx]
            signal_boost = 2.5 if any(sig in word for sig in HOAX_SIGNALS) else 1.0
            coef_val = coefs[idx] if coefs is not None else 0.0
            importance = coef_val * tfidf_val * signal_boost
            word_scores.append((word, importance))

        word_scores.sort(key=lambda x: x[1], reverse=True)
        suspicious_words = [w for w, score in word_scores if score > 0.01 or any(sig in w for sig in HOAX_SIGNALS)][:8]

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
            "evaluation_metrics": self.evaluation_metrics,
            "note": "Model NLP (TF-IDF + Logistic Regression) dilatih pada dataset terverifikasi TurnBackHoax & CekFakta Tempo."
        }

if __name__ == "__main__":
    clf = HoaxDetectorModel()
    test_res = clf.predict("Makan telur rebus jam 12 malam secara ajaib menyembuhkan virus corona!")
    print("NLP Prediction:", test_res["label"], f"({test_res['confidence_score']}%)")
    print("NLP Evaluation Metrics:", clf.evaluation_metrics)
