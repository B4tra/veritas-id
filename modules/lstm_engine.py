import numpy as np
import random
from modules.database import fetch_all_fact_checks
from modules.nlp_engine import preprocess_text, HOAX_SIGNALS

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
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
        
        self.train_model()
        
    def build_model(self):
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
        records = fetch_all_fact_checks()
        if not records:
            return
            
        texts = []
        labels = []
        for r in records:
            clean_t = preprocess_text(f"{r['title']} {r['claim']} {r['content']}")
            texts.append(clean_t)
            # 1 for HOAX, 0 for FAKTA
            labels.append(1 if r['label'] == 'HOAX' else 0)
            
        if len(set(labels)) < 2:
            return
            
        if TF_AVAILABLE:
            # Fit tokenizer
            self.tokenizer.fit_on_texts(texts)
            
            # Prepare sequences
            sequences = self.tokenizer.texts_to_sequences(texts)
            padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
            
            # Convert to numpy arrays
            X = np.array(padded_sequences)
            y = np.array(labels)
        
        # Build and train model
        self.model = self.build_model()
        if TF_AVAILABLE:
            # Train for a few epochs since dataset is small (for on-the-fly MVP)
            self.model.fit(X, y, epochs=5, verbose=0)
        self.is_trained = True
        
    def predict(self, raw_text):
        clean_input = preprocess_text(raw_text)
        
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
            # Provide mock random probabilities leaning towards text length heuristics if TF is missing
            hoax_prob = min(max(random.gauss(0.5, 0.2), 0.1), 0.9)
            if any(sig in clean_input for sig in HOAX_SIGNALS):
                hoax_prob = min(hoax_prob + 0.3, 0.95)
            fakta_prob = 1.0 - hoax_prob
        else:
            sequence = self.tokenizer.texts_to_sequences([clean_input])
            padded_sequence = pad_sequences(sequence, maxlen=self.max_length, padding='post', truncating='post')
            
            # Predict
            hoax_prob = float(self.model.predict(padded_sequence, verbose=0)[0][0])
            fakta_prob = 1.0 - hoax_prob
        
        if hoax_prob >= 0.53:
            label = "BERPOTENSI HOAX"
            badge_type = "danger"
            confidence = round(hoax_prob * 100, 1)
        elif fakta_prob >= 0.53:
            label = "KEMUNGKINAN FAKTA"
            badge_type = "success"
            confidence = round(fakta_prob * 100, 1)
        else:
            if hoax_prob > fakta_prob:
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
            "note": "Prediksi dari model Deep Learning LSTM." if TF_AVAILABLE else "Mode Simulasi (TensorFlow tidak terinstal). Menggunakan mock LSTM."
        }
