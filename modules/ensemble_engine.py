# Modul ensemble voting ini digunakan untuk menggabungkan 4 sinyal deteksi:
# NLP (25%), LSTM (20%), LLM (35%), dan Cross-Checker (20%).
# Skor gabungan akan menentukan hasil akhir klasifikasi (hoaks atau bukan).
"""
Ensemble Voting Engine for VERITAS-ID
Combines 4 detection signals into a unified final verdict & confidence score:
1. NLP Model (Logistic Regression + Heuristics) - Weight: 25%
2. LSTM Model (Deep Learning) - Weight: 20%
3. OpenRouter LLM Verdict (Zero-shot Reasoning) - Weight: 35%
4. Cross-Checker Similarity (Verified Fact Check Corpus) - Weight: 20%
"""

# Menghitung skor ensemble berdasarkan hasil prediksi 4 model yang berbeda
def compute_ensemble_verdict(nlp_res, lstm_res, llm_verdict_res, cross_matches):
    """
    Computes an integrated ensemble verdict score and final label.
    """
    # 1. NLP Model Score (Hoax Probability 0 - 100%)
    nlp_hoax_prob = nlp_res.get("hoax_prob", 50.0) if nlp_res else 50.0

    # 2. LSTM Model Score (Hoax Probability 0 - 100%)
    lstm_hoax_prob = lstm_res.get("hoax_prob", 50.0) if lstm_res else 50.0

    # 3. LLM Verdict Score
    llm_verdict_str = llm_verdict_res.get("verdict", "BELUM ADA BUKTI") if llm_verdict_res else "BELUM ADA BUKTI"
    llm_hoax_prob = llm_verdict_res.get("hoax_probability", 50.0) if llm_verdict_res else 50.0

    # 4. Cross-Checker Score
    cross_hoax_prob = 50.0
    top_match = cross_matches[0] if cross_matches else None
    if top_match:
        sim = top_match.get("similarity_score", 0.0)
        label = top_match.get("label", "")
        # Logika penyesuaian skor berdasarkan kesamaan referensi dengan database cek fakta
        if label == "HOAX":
            # High similarity to a known HOAX article -> High Hoax score
            cross_hoax_prob = min(50.0 + (sim * 0.5), 98.0)
        elif label == "FAKTA":
            # High similarity to a known FAKTA article -> Low Hoax score
            cross_hoax_prob = max(50.0 - (sim * 0.5), 5.0)

    # Weights
    # Bobot penilaian untuk setiap model pendeteksi
    w_nlp = 0.25
    w_lstm = 0.20
    w_llm = 0.35
    w_cross = 0.20

    # Ensemble Hoax Score Calculation (0 - 100%)
    # Perhitungan total skor hoaks dengan mengalikan probabilitas dan bobot tiap model
    ensemble_hoax_score = (
        (nlp_hoax_prob * w_nlp) +
        (lstm_hoax_prob * w_lstm) +
        (llm_hoax_prob * w_llm) +
        (cross_hoax_prob * w_cross)
    )

    ensemble_hoax_score = round(ensemble_hoax_score, 1)
    fakta_score = round(100.0 - ensemble_hoax_score, 1)

    # Determine final verdict category and badge
    # Menentukan label klasifikasi akhir berdasarkan threshold dari skor ensemble
    if ensemble_hoax_score >= 58.0:
        final_label = "BERPOTENSI HOAX"
        badge_type = "danger"
        confidence = ensemble_hoax_score
        verdict_summary = "Terdapat indikasi kuat bahwa informasi ini berpotensi hoaks/disinformasi."
    elif ensemble_hoax_score <= 42.0:
        final_label = "KEMUNGKINAN FAKTA"
        badge_type = "success"
        confidence = fakta_score
        verdict_summary = "Berdasarkan analisis ensemble multi-model, informasi ini cenderung akurat/fakta."
    else:
        final_label = "PERLU VERIFIKASI LANJUT"
        badge_type = "warning"
        confidence = max(ensemble_hoax_score, fakta_score)
        verdict_summary = "Hasil analisis menunjukkan ambigu/perlu konfirmasi lebih lanjut dari sumber resmi."

    return {
        "final_label": final_label,
        "badge_type": badge_type,
        "confidence_score": confidence,
        "hoax_score_percent": ensemble_hoax_score,
        "fakta_score_percent": fakta_score,
        "verdict_summary": verdict_summary,
        "llm_verdict": llm_verdict_str,
        "llm_reasoning": llm_verdict_res.get("reasoning", "") if llm_verdict_res else "",
        "breakdown": {
            "nlp": {"hoax_prob": nlp_hoax_prob, "weight": "25%", "label": nlp_res.get("label", "-") if nlp_res else "-"},
            "lstm": {"hoax_prob": lstm_hoax_prob, "weight": "20%", "label": lstm_res.get("label", "-") if lstm_res else "-"},
            "llm": {"hoax_prob": llm_hoax_prob, "weight": "35%", "verdict": llm_verdict_str},
            "cross_checker": {"hoax_prob": cross_hoax_prob, "weight": "20%", "top_match": top_match["title"] if top_match else "Tidak Ada Match"}
        }
    }
