# ==============================================================================
# Modul: lstm_engine.py
# Deskripsi: Mesin klasifikasi teks berbasis Deep Learning arsitektur LSTM:
#            1. Pelatihan model pada sekuens token data latih (train_dataset.csv)
#            2. Evaluasi performa pada data uji (Akurasi, Presisi, Recall, F1, Confusion Matrix)
#            3. Prediksi sekuensial probabilitas HOAX vs FAKTA (dengan Fallback Heuristik)
# Bagian dari: Tahap 4 - Modelling Dataset VERITAS-ID
# ==============================================================================

import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from modules.database import fetch_all_fact_checks
from modules.nlp_engine import preprocess_text, HOAX_SIGNALS, extract_text_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "data", "labeled", "train_dataset.csv")
TEST_CSV_PATH = os.path.join(BASE_DIR, "data", "labeled", "test_dataset.csv")

try:
    import tensorflow as tf  # type: ignore
    Sequential = tf.keras.models.Sequential
    Embedding = tf.keras.layers.Embedding
    LSTM = tf.keras.layers.LSTM
    Dense = tf.keras.layers.Dense
    Tokenizer = tf.keras.preprocessing.text.Tokenizer
    pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
    TF_AVAILABLE = True
except (ImportError, ModuleNotFoundError, AttributeError):
    TF_AVAILABLE = False

class LSTMDetectionModel:
    def __init__(self, vocab_size=5000, max_length=100, embedding_dim=32, lstm_units=32):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>") if TF_AVAILABLE else None
        self.model = None
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
        
    def build_model(self):
        """Menyusun arsitektur neural network LSTM (Embedding -> LSTM -> Dense -> Sigmoid)."""
        if not TF_AVAILABLE:
            return None
        model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_length),
            LSTM(self.lstm_units, return_sequences=False),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
        
    def train_model(self):
        """Memproses data latih dan melatih model LSTM, lalu mengevaluasi pada data uji."""
        train_texts = []
        train_labels = []

        # Prioritas 1: train_dataset.csv
        if os.path.exists(TRAIN_CSV_PATH):
            try:
                df_train = pd.read_csv(TRAIN_CSV_PATH)
                if not df_train.empty and "cleaned_text" in df_train.columns and "label_num" in df_train.columns:
                    train_texts = df_train["cleaned_text"].fillna("").tolist()
                    train_labels = df_train["label_num"].tolist()
            except Exception as e:
                print(f"[LSTM Model] Gagal membaca train_dataset.csv: {e}")

        # Fallback: Ambil dari SQLite
        if not train_texts or len(set(train_labels)) < 2:
            records = fetch_all_fact_checks()
            if records:
                train_texts = [preprocess_text(f"{r['title']} {r['claim']} {r['content']}") for r in records]
                train_labels = [1 if r['label'] == 'HOAX' else 0 for r in records]

        if len(set(train_labels)) < 2:
            return

        if TF_AVAILABLE and self.tokenizer is not None:
            self.tokenizer.fit_on_texts(train_texts)
            sequences = self.tokenizer.texts_to_sequences(train_texts)
            padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
            
            X_train = np.array(padded)
            y_train = np.array(train_labels)
            
            # Bobot kelas seimbang
            n_samples = len(y_train)
            n_hoax = sum(y_train)
            n_fakta = n_samples - n_hoax
            if n_hoax > 0 and n_fakta > 0:
                w_hoax = n_samples / (2.0 * n_hoax)
                w_fakta = n_samples / (2.0 * n_fakta)
                sample_weights = np.array([w_hoax if lbl == 1 else w_fakta for lbl in y_train])
            else:
                sample_weights = np.ones(n_samples)
            
            self.model = self.build_model()
            if self.model is not None:
                self.model.fit(X_train, y_train, epochs=8, verbose=0, sample_weight=sample_weights)

        self.is_trained = True
        self._evaluate_on_test_set(train_count=len(train_texts))

    def _evaluate_on_test_set(self, train_count=0):
        """Evaluasi metrik LSTM pada test dataset terpisah."""
        test_texts = []
        test_labels = []

        if os.path.exists(TEST_CSV_PATH):
            try:
                df_test = pd.read_csv(TEST_CSV_PATH)
                if not df_test.empty and "cleaned_text" in df_test.columns and "label_num" in df_test.columns:
                    test_texts = df_test["cleaned_text"].fillna("").tolist()
                    test_labels = df_test["label_num"].tolist()
            except Exception as e:
                print(f"[LSTM Model] Gagal membaca test_dataset.csv: {e}")

        if not test_texts:
            test_texts = [preprocess_text(f"{r['title']} {r['claim']} {r['content']}") for r in fetch_all_fact_checks()][:4]
            test_labels = [1 if r['label'] == 'HOAX' else 0 for r in fetch_all_fact_checks()][:4]

        if test_texts and len(set(test_labels)) >= 1:
            y_pred = []
            if TF_AVAILABLE and self.model is not None and self.tokenizer is not None:
                seqs = self.tokenizer.texts_to_sequences(test_texts)
                pad = pad_sequences(seqs, maxlen=self.max_length, padding='post', truncating='post')
                preds_prob = self.model.predict(pad, verbose=0)
                y_pred = [1 if float(p[0]) >= 0.50 else 0 for p in preds_prob]
            else:
                # Estimasi akurasi heuristik
                for t in test_texts:
                    _, _, sigs, _ = extract_text_features(t)
                    y_pred.append(1 if sigs >= 1 else 0)

            acc = round(accuracy_score(test_labels, y_pred) * 100, 1)
            prec = round(precision_score(test_labels, y_pred, pos_label=1, zero_division=0) * 100, 1)
            rec = round(recall_score(test_labels, y_pred, pos_label=1, zero_division=0) * 100, 1)
            f1 = round(f1_score(test_labels, y_pred, pos_label=1, zero_division=0) * 100, 1)
            cm = confusion_matrix(test_labels, y_pred, labels=[0, 1])

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
        clean_input = preprocess_text(raw_text)
        
        if len(clean_input.split()) < 2:
            return {
                "label": "TIDAK CUKUP DATA",
                "badge_type": "warning",
                "confidence_score": 0.0,
                "hoax_prob": 0.0,
                "fakta_prob": 0.0,
                "note": "Input terlalu pendek untuk dianalisis oleh model LSTM."
            }
            
        if not self.is_trained:
            self.train_model()
            
        if not TF_AVAILABLE or self.model is None or self.tokenizer is None:
            # Fallback analisis heuristik
            excl_count, caps_ratio, signal_count, url_count = extract_text_features(raw_text)
            hoax_prob = 0.45
            if signal_count >= 3:
                hoax_prob += 0.25
            elif signal_count >= 2:
                hoax_prob += 0.15
            elif signal_count >= 1:
                hoax_prob += 0.08
            
            if excl_count >= 3:
                hoax_prob += 0.05
            if caps_ratio > 0.3:
                hoax_prob += 0.05
            if url_count >= 1:
                hoax_prob += 0.03
            
            hoax_prob = min(max(hoax_prob, 0.05), 0.95)
            fakta_prob = 1.0 - hoax_prob
        else:
            sequence = self.tokenizer.texts_to_sequences([clean_input])
            padded_sequence = pad_sequences(sequence, maxlen=self.max_length, padding='post', truncating='post')
            
            hoax_prob = float(self.model.predict(padded_sequence, verbose=0)[0][0])
            fakta_prob = 1.0 - hoax_prob
        
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
            "evaluation_metrics": self.evaluation_metrics,
            "note": "Model Deep Learning LSTM dilatih pada dataset terverifikasi TurnBackHoax & CekFakta Tempo." if TF_AVAILABLE else "Mode Simulasi Heuristik (TensorFlow tidak terinstal). Menggunakan analisis sekuens kata kunci."
        }

if __name__ == "__main__":
    lstm = LSTMDetectionModel()
    pred = lstm.predict("Pemerintah berikan bantuan tunai 600 ribu lewat wa bit.ly")
    print("LSTM Prediction:", pred["label"], f"({pred['confidence_score']}%)")
    print("LSTM Evaluation Metrics:", lstm.evaluation_metrics)
