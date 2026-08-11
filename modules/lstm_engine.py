# Modul ini merupakan mesin klasifikasi teks berbasis Deep Learning menggunakan arsitektur LSTM.
# Model ini dirancang untuk mempelajari konteks dan urutan kata dalam mendeteksi pola hoax.
# Jika TensorFlow tidak tersedia, modul ini otomatis melakukan fallback ke analisis heuristik teks.
import numpy as np
import random
from modules.database import fetch_all_fact_checks
from modules.nlp_engine import preprocess_text, HOAX_SIGNALS, extract_text_features

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

# Kelas utama untuk mengelola siklus hidup model LSTM dalam mendeteksi hoax
class LSTMDetectionModel:
    def __init__(self, vocab_size=5000, max_length=100, embedding_dim=32, lstm_units=32):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>") if TF_AVAILABLE else None
        self.model = None
        self.is_trained = False
        
        self.train_model()
        
    # Fungsi untuk menyusun arsitektur model (Embedding -> LSTM -> Dense)
    def build_model(self):
        if not TF_AVAILABLE:
            return None
        # Membangun lapisan neural network untuk memproses teks secara sekuensial
        model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_length),
            LSTM(self.lstm_units, return_sequences=False),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
        
    # Fungsi untuk memproses data dari database dan melatih model LSTM
    def train_model(self):
        records = fetch_all_fact_checks()
        if not records:
            return
            
        texts = []
        labels = []
        # Menyiapkan teks masukan beserta label target yang sesuai
        for r in records:
            clean_t = preprocess_text(f"{r['title']} {r['claim']} {r['content']}")
            texts.append(clean_t)
            # 1 for HOAX, 0 for FAKTA
            labels.append(1 if r['label'] == 'HOAX' else 0)
            
        if len(set(labels)) < 2:
            return
            
        if TF_AVAILABLE:
            # Mengonversi kumpulan teks menjadi representasi sekuens angka
            # Fit tokenizer
            self.tokenizer.fit_on_texts(texts)
            
            # Prepare sequences
            sequences = self.tokenizer.texts_to_sequences(texts)
            padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
            
            # Convert to numpy arrays
            X = np.array(padded_sequences)
            y = np.array(labels)
            
            # Menghitung bobot kelas seimbang agar model tidak bias pada label yang lebih banyak
            # Compute balanced class weights as sample weights
            n_samples = len(y)
            n_hoax = sum(y)
            n_fakta = n_samples - n_hoax
            if n_hoax > 0 and n_fakta > 0:
                weight_hoax = n_samples / (2.0 * n_hoax)
                weight_fakta = n_samples / (2.0 * n_fakta)
                sample_weights = np.array([weight_hoax if label == 1 else weight_fakta for label in y])
            else:
                sample_weights = np.ones(n_samples)
        
        # Build and train model
        self.model = self.build_model()
        if TF_AVAILABLE:
            # Melatih model dengan mempertimbangkan bobot kelas sampel (balanced class weights)
            # Train with balanced sample weights
            self.model.fit(X, y, epochs=8, verbose=0, sample_weight=sample_weights)
        self.is_trained = True
        
    # Fungsi untuk melakukan prediksi terhadap input teks terbaru
    def predict(self, raw_text):
        clean_input = preprocess_text(raw_text)
        
        # Mengecek apakah teks memiliki jumlah kata yang cukup untuk diproses model
        if len(clean_input.split()) < 3:
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
            
        if not TF_AVAILABLE:
            # Fallback (skenario pengganti) dengan menggunakan heuristik saat TensorFlow tidak terinstal
            # Heuristic-based mock prediction (NOT random)
            excl_count, caps_ratio, signal_count, url_count = extract_text_features(raw_text)
            
            # Memulai perhitungan probabilitas dari tingkat menengah (netral)
            # Start with neutral probability
            hoax_prob = 0.45
            
            # Menambahkan probabilitas berdasarkan temuan kata-kata sinyal hoax
            # Boost based on hoax signals
            if signal_count >= 3:
                hoax_prob += 0.25
            elif signal_count >= 2:
                hoax_prob += 0.15
            elif signal_count >= 1:
                hoax_prob += 0.08
            
            # Menambahkan probabilitas berdasarkan indikator gaya tulisan sensasional
            # Boost for sensationalism indicators
            if excl_count >= 3:
                hoax_prob += 0.05
            if caps_ratio > 0.3:
                hoax_prob += 0.05
            if url_count >= 1:
                hoax_prob += 0.03
            
            # Add small randomness for variation
            hoax_prob += random.gauss(0, 0.03)
            hoax_prob = min(max(hoax_prob, 0.05), 0.95)
            fakta_prob = 1.0 - hoax_prob
        else:
            # Melakukan prediksi menggunakan model LSTM yang telah dilatih
            sequence = self.tokenizer.texts_to_sequences([clean_input])
            padded_sequence = pad_sequences(sequence, maxlen=self.max_length, padding='post', truncating='post')
            
            # Predict
            hoax_prob = float(self.model.predict(padded_sequence, verbose=0)[0][0])
            fakta_prob = 1.0 - hoax_prob
        
        # Menggolongkan hasil klasifikasi dengan batas prediksi probabilitas 50%
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
            "note": "Prediksi dari model Deep Learning LSTM (class_weight=balanced)." if TF_AVAILABLE else "Mode Simulasi Heuristik (TensorFlow tidak terinstal). Menggunakan analisis sinyal kata kunci."
        }
