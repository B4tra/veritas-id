# =============================================================
# Modul: app.py
# Deskripsi: File utama aplikasi VERITAS-ID berbasis Streamlit.
#            Menyediakan antarmuka visual terpadu untuk:
#            - Pengecekan klaim / deteksi hoax multi-sinyal (NLP, LSTM, LLM, Cross-Checker)
#            - Visualisasi 4 Tahapan Pipeline Data (1. Scraping Mentah, 2. Preprocessing, 3. Labelling & Split, 4. Modeling & Benchmark)
#            - Basis Data Rujukan Cek Fakta (TurnBackHoax & Tempo)
#            - Admin Panel untuk eksekusi pipeline runtut 1-4 & konfigurasi API/Email
# Bagian dari: VERITAS-ID - Sistem Deteksi Hoax Multi-Sumber Indonesia
# =============================================================

import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Inisialisasi database dengan data seed awal jika belum ada
from modules.seed_data import init_database, DB_PATH
init_database(DB_PATH)

# Import modul-modul internal VERITAS-ID
from modules.database import save_search_history, get_recent_history, fetch_all_fact_checks
from modules.url_parser import is_valid_url, extract_content_from_url
from modules.scraper_engine import run_scraping_pipeline, RAW_CSV_PATH
from modules.preprocessor import run_preprocessing_pipeline, preprocess_text_single, PROCESSED_CSV_PATH, INDONESIAN_STOPWORDS
from modules.labelling_engine import run_labelling_pipeline, get_label_distribution, LABELED_CSV_PATH, TRAIN_CSV_PATH, TEST_CSV_PATH
from modules.nlp_engine import HoaxDetectorModel, HOAX_SIGNALS
from modules.lstm_engine import LSTMDetectionModel
from modules.cross_checker import CrossChecker
from modules.llm_extractor import extract_claim_with_llm, test_openrouter_connection, get_llm_verdict, DEFAULT_OPENROUTER_MODEL
from modules.ensemble_engine import compute_ensemble_verdict, get_model_benchmark_table
from modules.email_service import send_test_email

def plot_confusion_matrix_heatmap(cm, model_name="Model", cmap="Blues"):
    """Menghasilkan visualisasi Heatmap Confusion Matrix ringkas dan kontras tinggi."""
    fig, ax = plt.subplots(figsize=(2.7, 2.2), dpi=140)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    cm_arr = np.array(cm)
    im = ax.imshow(cm_arr, interpolation='nearest', cmap=cmap)
    
    classes = ["FAKTA (0)", "HOAX (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, color='#e2e8f0', fontsize=8, fontweight='bold')
    ax.set_yticklabels(classes, color='#e2e8f0', fontsize=8, fontweight='bold')

    # Garis pemisah antar sel (grid)
    ax.set_xticks(np.arange(len(classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(classes) + 1) - .5, minor=True)
    ax.grid(which='minor', color='#0f172a', linestyle='-', linewidth=2.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    for i in range(len(classes)):
        for j in range(len(classes)):
            val = int(cm_arr[i][j])
            # Warna dinamis adaptif berbasis luminance agar teks selalu kontras tinggi:
            # Teks bold hitam pekat (#000000) pada latar terang, dan bold putih (#ffffff) pada latar gelap.
            rgba = im.cmap(im.norm(val))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = '#000000' if luminance > 0.55 else '#ffffff'
            ax.text(j, i, f"{val}",
                    ha="center", va="center",
                    color=text_color, fontsize=12, fontweight='bold')

    ax.set_ylabel('Aktual (Ground Truth)', color='#94a3b8', fontsize=7.5, fontweight='bold', labelpad=4)
    ax.set_xlabel('Prediksi (Predicted)', color='#94a3b8', fontsize=7.5, fontweight='bold', labelpad=4)
    ax.set_title(f'{model_name}', color='#ffffff', fontsize=8.5, fontweight='bold', pad=7)

    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

    plt.tight_layout(pad=0.6)
    return fig

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="VERITAS-ID | Sistem Deteksi Hoax NLP & Pipeline Data Indonesia",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Memuat file CSS kustom dari assets/style.css
CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==========================================
# INISIALISASI SESSION STATE
# ==========================================
if "history" not in st.session_state:
    st.session_state.history = []
if "nlp_model" not in st.session_state:
    st.session_state.nlp_model = HoaxDetectorModel()
if "lstm_model" not in st.session_state:
    st.session_state.lstm_model = LSTMDetectionModel()
if "cross_checker" not in st.session_state:
    st.session_state.cross_checker = CrossChecker()

# Pastikan file CSV dasar 1-3 terbuat jika belum ada
if not os.path.exists(RAW_CSV_PATH) or not os.path.exists(PROCESSED_CSV_PATH) or not os.path.exists(LABELED_CSV_PATH):
    try:
        run_preprocessing_pipeline()
        run_labelling_pipeline()
    except Exception:
        pass

# ==========================================
# HEADER UTAMA APLIKASI
# ==========================================
st.markdown("""
    <div class="main-header">
        <h1>🛡️ VERITAS-ID</h1>
        <p>Sistem Deteksi Hoax Multi-Sinyal & Pipeline Data Cek Fakta Terverifikasi Indonesia<br>
        <small style="color: #94a3b8; font-weight: 500;">(TurnBackHoax.id / MAFINDO [Fokus Hoax] • CekFakta Tempo [Fokus Fakta])</small></p>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# NAVIGASI TAB UTAMA
# ==========================================
tab_detect, tab_pipeline, tab_database, tab_admin, tab_about = st.tabs([
    "🛡️ Pengecekan Klaim",
    "📊 Pipeline & Evaluasi Dataset",
    "📚 Basis Data Rujukan",
    "⚙️ Admin Panel Retrain",
    "ℹ️ Tentang & PRD"
])

# ==========================================================
# TAB 1: PENGECEKAN KLAIM (LIVE MULTI-SIGNAL INFERENCE)
# ==========================================================
with tab_detect:
    st.markdown("### 🔍 Analisis Klaim Berita & Deteksi Hoax")
    st.caption("Masukkan naskah klaim, pesan berantai WhatsApp, atau tempelkan URL artikel berita untuk dianalisis oleh 4 sinyal AI.")

    user_input = st.text_area(
        label="Kotak Input Klaim/Tautan",
        value=st.session_state.get("current_input", ""),
        placeholder="Tempel teks klaim atau tautan berita di sini (Contoh: 'Makan telur rebus jam 12 malam secara ajaib menyembuhkan Corona' atau 'https://cekfakta.tempo.co/...').",
        height=130,
        label_visibility="collapsed"
    )

    if user_input != st.session_state.get("current_input", ""):
        st.session_state.current_input = user_input

    col_btn1, col_btn2, col_btn3 = st.columns([2, 1.2, 1.2])

    with col_btn1:
        if st.button("🚀 Langkah 1: Ekstrak Klaim (OpenRouter AI)", type="primary", use_container_width=True):
            if not user_input.strip():
                st.warning("⚠️ Mohon masukkan teks klaim atau tautan berita terlebih dahulu.")
            else:
                input_text = user_input.strip()
                is_url_input = is_valid_url(input_text)
                raw_text_for_llm = input_text

                if is_url_input:
                    st.toast("Tautan berita terdeteksi! Mengekstraksi konten artikel...", icon="ℹ️")
                    url_res = extract_content_from_url(input_text)
                    if url_res["success"]:
                        raw_text_for_llm = url_res["text"]

                CONFIG_PATH = os.path.join(os.path.dirname(__file__), "data", "config.json")
                api_key = ""
                model_name = DEFAULT_OPENROUTER_MODEL
                if os.path.exists(CONFIG_PATH):
                    try:
                        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                            api_key = cfg.get("openrouter_api_key", cfg.get("llm_api_key", ""))
                            model_name = cfg.get("openrouter_model", DEFAULT_OPENROUTER_MODEL)
                    except:
                        pass

                with st.spinner("Mengekstraksi klaim inti & entitas teks via AI..."):
                    llm_res = extract_claim_with_llm(raw_text_for_llm, api_key=api_key, model_name=model_name)
                    st.session_state.llm_result = llm_res
                    st.session_state.edited_claim = llm_res["extracted_claim"]
                    st.session_state["edited_claim_input"] = llm_res["extracted_claim"]
                    st.session_state.show_extraction_card = True
                    st.session_state.show_results = False

    with col_btn2:
        if st.button("💡 Sampel Hoax", use_container_width=True):
            st.session_state.current_input = "Beredar pesan berantai di WhatsApp yang mengklaim bahwa memakan telur rebus pada jam 12 malam secara ajaib dapat menangkal dan menyembuhkan infeksi virus Corona."
            st.session_state.show_extraction_card = False
            st.session_state.show_results = False
            st.rerun()

    with col_btn3:
        if st.button("💡 Sampel Fakta", use_container_width=True):
            st.session_state.current_input = "Kejaksaan Agung bersama Badan Pengawasan Keuangan dan Pembangunan (BPKP) merilis audit komprehensif mengenai estimasi kerugian keuangan negara pada perkara tata niaga timah."
            st.session_state.show_extraction_card = False
            st.session_state.show_results = False
            st.rerun()

    # KARTU REVIEW HASIL EKSTRAKSI KLAIM LLM
    if st.session_state.get("show_extraction_card", False) and "llm_result" in st.session_state:
        res = st.session_state.llm_result
        st.markdown("---")
        st.markdown("### 📋 Hasil Ekstraksi Klaim Inti & Entitas")

        if res.get("api_key_missing", False):
            st.warning(f"{res['message']} (Anda dapat memasukkan API Key di Tab ⚙️ Admin Panel).")
        elif not res.get("is_llm", False):
            st.info(f"{res['message']}")
        else:
            st.success(f"Ekstraksi klaim sukses via OpenRouter AI (`{res.get('model_used', DEFAULT_OPENROUTER_MODEL)}`)")

        col_c1, col_c2 = st.columns([1.8, 1.2])
        with col_c1:
            st.markdown("#### Tinjau & Edit Kalimat Klaim Utama:")
            edited_text = st.text_area(
                label="Kalimat Klaim Utama:",
                value=st.session_state.get("edited_claim", res["extracted_claim"]),
                height=90,
                key="edited_claim_input",
                help="Kalimat klaim ini yang akan dikirimkan ke 4 sinyal deteksi."
            )
            st.session_state.edited_claim = edited_text

        with col_c2:
            st.markdown("#### Informasi Tambahan Artikel:")
            st.write(f"**Ringkasan:** {res['summary']}")
            if res.get("key_entities"):
                tags_html = " ".join([f'<span class="word-tag-green">{e}</span>' for e in res["key_entities"]])
                st.markdown(f"**Entitas:** {tags_html}", unsafe_allow_html=True)
            st.caption(f"Tingkat Sensasionalisme: **{res.get('sensational_rating', 'RENDAH')}**")

        st.write("")
        if st.button("🛡️ Langkah 2: Verifikasi Fakta 4 Sinyal Sekarang", type="primary", use_container_width=True):
            st.session_state.show_results = True

    # HASIL VERIFIKASI MULTI-SINYAL
    if st.session_state.get("show_results", False):
        analysis_text = st.session_state.get("edited_claim", st.session_state.get("current_input", "")).strip()
        input_text = st.session_state.get("current_input", "").strip()
        is_url_input = is_valid_url(input_text)

        with st.spinner("Memproses 4 Sinyal Deteksi (NLP LR + LSTM + OpenRouter LLM + Cross-Checker)..."):
            prediction = st.session_state.nlp_model.predict(analysis_text)
            prediction_lstm = st.session_state.lstm_model.predict(analysis_text)

            CONFIG_PATH = os.path.join(os.path.dirname(__file__), "data", "config.json")
            api_key = ""
            model_name = DEFAULT_OPENROUTER_MODEL
            if os.path.exists(CONFIG_PATH):
                try:
                    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                        api_key = cfg.get("openrouter_api_key", cfg.get("llm_api_key", ""))
                        model_name = cfg.get("openrouter_model", DEFAULT_OPENROUTER_MODEL)
                except:
                    pass

            llm_verdict_res = get_llm_verdict(analysis_text, api_key=api_key, model_name=model_name)
            matches = st.session_state.cross_checker.find_matches(analysis_text)
            top_match = matches[0] if matches else None

            ensemble_res = compute_ensemble_verdict(prediction, prediction_lstm, llm_verdict_res, matches)

            save_search_history(
                input_text=analysis_text,
                input_type="url" if is_url_input else "text",
                predicted_label=ensemble_res["final_label"],
                confidence_score=ensemble_res["confidence_score"],
                matched_reference_id=top_match["id"] if top_match else None
            )

        st.markdown("---")
        st.markdown("## 🎯 Hasil Vonis Ensemble Multi-Model")

        ens_badge_class = f"badge-{ensemble_res['badge_type']}"
        verdict_badge_colors = {
            "BENAR": "#10b981",
            "SEBAGIAN BENAR": "#06b6d4",
            "BELUM ADA BUKTI": "#64748b",
            "SESAT": "#f59e0b",
            "KELIRU": "#ef4444"
        }
        v_color = verdict_badge_colors.get(ensemble_res["llm_verdict"], "#64748b")

        col_ens1, col_ens2 = st.columns([1.3, 1.7])
        with col_ens1:
            st.markdown("#### Verdict Akhir Ensemble")
            st.markdown(f'<div class="badge-result {ens_badge_class}">{ensemble_res["final_label"]}</div>', unsafe_allow_html=True)
            st.write("")
            st.markdown(f"**Ringkasan:** {ensemble_res['verdict_summary']}")

            st.markdown(f"""
                <div style="margin-top: 1rem; padding: 0.9rem; border-radius: 8px; background: rgba(255,255,255,0.07); border-left: 4px solid {v_color};">
                    <small style="color: #cbd5e1; font-weight: bold; letter-spacing: 0.5px;">VONIS CEK FAKTA LLM:</small><br>
                    <span style="font-size: 1.15rem; font-weight: bold; color: {v_color};">🏷️ {ensemble_res['llm_verdict']}</span><br>
                    <p style="color: #ffffff; font-size: 0.95rem; margin-top: 0.4rem; margin-bottom: 0; line-height: 1.5;"><em>"{ensemble_res['llm_reasoning']}"</em></p>
                </div>
            """, unsafe_allow_html=True)

        with col_ens2:
            st.markdown("#### Indikator Risiko Hoaks (Ensemble Score)")
            hoax_pct = ensemble_res["hoax_score_percent"]
            st.progress(hoax_pct / 100.0)

            em1, em2, em3 = st.columns(3)
            em1.metric("Skor Risiko Hoaks", f"{hoax_pct}%")
            em2.metric("Keandalan Fakta", f"{ensemble_res['fakta_score_percent']}%")
            em3.metric("Status Vonis LLM", ensemble_res["llm_verdict"])

            st.markdown("##### Matriks Kontribusi Sinyal Ensemble:")
            breakdown = ensemble_res["breakdown"]
            st.markdown(f"""
            | Sinyal Deteksi | Bobot | Hasil Prediksi | Probabilitas Hoaks |
            | :--- | :---: | :--- | :---: |
            | **NLP (Logistic Regression)** | 25% | `{breakdown['nlp']['label']}` | **{breakdown['nlp']['hoax_prob']}%** |
            | **LSTM (Deep Learning)** | 20% | `{breakdown['lstm']['label']}` | **{breakdown['lstm']['hoax_prob']}%** |
            | **OpenRouter LLM (Zero-shot)** | 35% | `{breakdown['llm']['verdict']}` | **{breakdown['llm']['hoax_prob']}%** |
            | **Cross-Checker (Corpus Match)** | 20% | `{breakdown['cross_checker']['top_match'][:32]}...` | **{breakdown['cross_checker']['hoax_prob']}%** |
            """)

        if prediction.get("suspicious_words"):
            st.markdown("##### ⚠️ Kata / Frasa Sensasional Terdeteksi (Explainability):")
            tags_html = "".join([f'<span class="word-tag">⚠️ {word}</span>' for word in prediction["suspicious_words"]])
            st.markdown(f'<div style="margin-bottom: 1rem;">{tags_html}</div>', unsafe_allow_html=True)

        # RUJUKAN ARTIKEL CEK FAKTA
        st.markdown("---")
        st.markdown("### 📚 Rujukan Verifikasi Silang (TurnBackHoax & Tempo)")
        if matches:
            st.success(f"Ditemukan {len(matches)} artikel rujukan terverifikasi:")
            for m in matches:
                ref_type = "hoax" if m["label"] == "HOAX" else "fakta"
                st.markdown(f"""
                    <div class="ref-card {ref_type}">
                        <h4><a href="{m['source_url']}" target="_blank" style="text-decoration: none; color: inherit;">🔗 {m['title']}</a></h4>
                        <p><strong>Sumber Resmi:</strong> {m['source_name']} | <strong>Kategori:</strong> {m['category']} | <strong>Kemiripan Teks:</strong> {m['similarity_score']}%</p>
                        <p style="background: rgba(0,0,0,0.1); padding: 0.8rem; border-radius: 6px;"><strong>📌 Vonis & Bukti:</strong><br>{m['verdict_details']}</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Tidak ditemukan klaim serupa di korpus rujukan. Prediksi murni didasarkan pada klasifikasi model NLP, LSTM, dan penalaran LLM.")

# ==========================================================
# TAB 2: PIPELINE & EVALUASI DATASET (TAHAP 1 - 4 RUNTUT)
# ==========================================================
with tab_pipeline:
    st.markdown("""
        <div class="step-banner">
            <h3>📊 Alur Pipeline Data & Pemodelan AI (Tahapan 1 s.d. 4)</h3>
            <p>Eksplorasi runtut setiap proses penambangan data: dari penarikan data mentah website, pembersihan teks, pelabelan biner, hingga evaluasi akurasi model.</p>
        </div>
    """, unsafe_allow_html=True)

    subtab_s1, subtab_s2, subtab_s3, subtab_s4 = st.tabs([
        "📥 Tahap 1: Scraping Mentah",
        "🧹 Tahap 2: Preprocessing Teks",
        "🏷️ Tahap 3: Labelling & Dataset Split",
        "🤖 Tahap 4: Benchmark & Evaluasi Model"
    ])

    # ----------------------------------------------------------
    # SUB-TAB 1: SCRAPING MENTAH
    # ----------------------------------------------------------
    with subtab_s1:
        st.markdown("### Tahap 1: Scraping / Crawling Dataset Mentah")
        st.caption("Pengumpulan artikel berita mentah secara real-time dari TurnBackHoax.id (fokus HOAX) dan CekFakta Tempo (fokus FAKTA).")

        df_raw = None
        if os.path.exists(RAW_CSV_PATH):
            try:
                df_raw = pd.read_csv(RAW_CSV_PATH)
            except Exception:
                pass

        if df_raw is None or df_raw.empty:
            conn = sqlite3.connect(DB_PATH)
            df_raw = pd.read_sql_query("SELECT id, source_name as source_platform, title, claim, content as raw_content, label, category, source_url, published_at, created_at as crawled_at FROM fact_checks", conn)
            conn.close()

        total_raw = len(df_raw) if df_raw is not None else 0
        tbh_raw_count = len(df_raw[df_raw["source_platform"].str.contains("TurnBackHoax|MAFINDO", case=False, na=False)]) if df_raw is not None else 0
        tempo_raw_count = len(df_raw[df_raw["source_platform"].str.contains("Tempo", case=False, na=False)]) if df_raw is not None else 0

        c_r1, c_r2, c_r3, c_r4 = st.columns(4)
        c_r1.metric("Total Data Mentah Terkumpul", total_raw)
        c_r2.metric("TurnBackHoax.id (HOAX)", tbh_raw_count)
        c_r3.metric("CekFakta Tempo (FAKTA)", tempo_raw_count)
        c_r4.metric("Format File Penyimpanan", "CSV (data/raw/)")

        st.markdown("#### Tabel Dataset Mentah (`data/raw/raw_scraped_dataset.csv`):")
        if df_raw is not None and not df_raw.empty:
            st.dataframe(
                df_raw,
                column_config={
                    "id": st.column_config.NumberColumn("ID", format="%d"),
                    "source_platform": "Sumber Platform",
                    "title": "Judul Artikel Mentah",
                    "raw_content": "Konten Berita Asli",
                    "label": "Label Awal",
                    "source_url": st.column_config.LinkColumn("Tautan Asli"),
                    "crawled_at": "Waktu Scraping"
                },
                hide_index=True,
                use_container_width=True
            )

            # Tombol Download CSV Mentah
            csv_data = df_raw.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Unduh Dataset Mentah (raw_scraped_dataset.csv)",
                data=csv_data,
                file_name="raw_scraped_dataset.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.warning("Belum ada dataset mentah. Silakan jalankan scraper di Tab ⚙️ Admin Panel.")

    # ----------------------------------------------------------
    # SUB-TAB 2: PREPROCESSING TEKS
    # ----------------------------------------------------------
    with subtab_s2:
        st.markdown("### Tahap 2: Preprocessing Teks Bahasa Indonesia")
        st.caption("Pembersihan teks terstruktur: Case Folding, Penghapusan URL & Simbol Spesial, Normalisasi Angka/Rupiah/Persen, Penghapusan Stopwords Bahasa Indonesia, dan Tokenisasi Kata.")

        df_proc = None
        if os.path.exists(PROCESSED_CSV_PATH):
            try:
                df_proc = pd.read_csv(PROCESSED_CSV_PATH)
            except Exception:
                pass

        if df_proc is not None and not df_proc.empty:
            avg_red = round(df_proc["reduction_percentage"].mean(), 1)
            total_proc = len(df_proc)
            total_before_tokens = df_proc["token_count_before"].sum()
            total_after_tokens = df_proc["token_count_after"].sum()

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Total Artikel Diproses", total_proc)
            p2.metric("Total Token Sebelum Pembersihan", f"{total_before_tokens:,}")
            p3.metric("Total Token Bersih (Cleaned)", f"{total_after_tokens:,}")
            p4.metric("Rata-rata Reduksi Noise", f"{avg_red}%")

            st.markdown("---")
            st.markdown("#### Perbandingan Teks: Sebelum vs Sesudah Preprocessing (Before vs After)")
            
            # Selector artikel interaktif untuk melihat before vs after
            article_options = [f"{row['id']}: {row['original_title'][:70]}..." for _, row in df_proc.iterrows()]
            selected_art_idx = st.selectbox("Pilih Artikel untuk Inspeksi Transformasi Teks:", range(len(article_options)), format_func=lambda x: article_options[x])

            sel_row = df_proc.iloc[selected_art_idx]
            
            col_b, col_a = st.columns(2)
            with col_b:
                st.markdown("""
                    <div class="compare-box" style="border-left: 4px solid #ef4444;">
                        <div class="compare-box-header">🔴 SEBELUM (RAW ORIGINAL TEXT)</div>
                    </div>
                """, unsafe_allow_html=True)
                st.text_area("Teks Asli:", value=sel_row["raw_text"], height=160, disabled=True, key="raw_view")
                st.caption(f"Jumlah Kata: **{sel_row['token_count_before']} kata**")

            with col_a:
                st.markdown("""
                    <div class="compare-box" style="border-left: 4px solid #10b981;">
                        <div class="compare-box-header">🟢 SESUDAH (CLEANED & TOKENIZED)</div>
                    </div>
                """, unsafe_allow_html=True)
                st.text_area("Teks Bersih:", value=sel_row["cleaned_text"], height=160, disabled=True, key="clean_view")
                st.caption(f"Jumlah Token Bersih: **{sel_row['token_count_after']} token** (Reduksi: **{sel_row['reduction_percentage']}%**)")

            st.markdown("##### Token Kata yang Dihasilkan:")
            tokens_list = str(sel_row["tokens"]).split(",") if pd.notna(sel_row["tokens"]) else []
            token_tags = " ".join([f'<span class="word-tag-green">{tok}</span>' for tok in tokens_list[:25]])
            st.markdown(f'<div style="line-height: 2;">{token_tags}</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### Tabel Lengkap Hasil Preprocessing (`data/processed/preprocessed_dataset.csv`):")
            st.dataframe(
                df_proc[['id', 'original_title', 'token_count_before', 'token_count_after', 'reduction_percentage', 'cleaned_text', 'label']],
                column_config={
                    "id": "ID",
                    "original_title": "Judul Asli",
                    "token_count_before": "Token Awal",
                    "token_count_after": "Token Bersih",
                    "reduction_percentage": "Reduksi (%)",
                    "cleaned_text": "Teks Hasil Preprocessing",
                    "label": "Label"
                },
                hide_index=True,
                use_container_width=True
            )

            # Tombol Download Preprocessed CSV
            proc_csv_data = df_proc.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Unduh Dataset Preprocessed (preprocessed_dataset.csv)",
                data=proc_csv_data,
                file_name="preprocessed_dataset.csv",
                mime="text/csv"
            )
        else:
            st.warning("Belum ada dataset preprocessed. Silakan jalankan pipeline di Tab ⚙️ Admin Panel.")

    # ----------------------------------------------------------
    # SUB-TAB 3: LABELLING & DATASET SPLIT
    # ----------------------------------------------------------
    with subtab_s3:
        st.markdown("### Tahap 3: Labelling & Stratified Train/Test Split")
        st.caption("Pelabelan biner ('HOAX' = 1, 'FAKTA' = 0) dan pembagian dataset menjadi 80% Data Latih (Train) dan 20% Data Uji (Test) secara terstratifikasi.")

        lbl_stats = get_label_distribution()
        df_train = None
        df_test = None
        if os.path.exists(TRAIN_CSV_PATH):
            try:
                df_train = pd.read_csv(TRAIN_CSV_PATH)
            except Exception:
                pass
        if os.path.exists(TEST_CSV_PATH):
            try:
                df_test = pd.read_csv(TEST_CSV_PATH)
            except Exception:
                pass

        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Total Dataset Berlabel", lbl_stats.get("total", 0))
        l2.metric("Kelas HOAX (1)", f"{lbl_stats.get('hoax_count', 0)} ({lbl_stats.get('hoax_pct', 0)}%)")
        l3.metric("Kelas FAKTA (0)", f"{lbl_stats.get('fakta_count', 0)} ({lbl_stats.get('fakta_pct', 0)}%)")
        l4.metric("Pembagian Rasio Data", "80% Train : 20% Test")

        st.markdown("#### Proporsi Keseimbangan Kelas Label:")
        hoax_pct_val = lbl_stats.get("hoax_pct", 50.0)
        st.progress(hoax_pct_val / 100.0)
        st.caption(f"Distribusi Kelas: 🔴 **HOAX**: {hoax_pct_val}% | 🟢 **FAKTA**: {round(100.0 - hoax_pct_val, 1)}%")

        st.markdown("---")
        col_tr, col_te = st.columns(2)
        with col_tr:
            st.markdown(f"#### 1. Data Latih / Training Set 80% ({len(df_train) if df_train is not None else 0} Sampel)")
            if df_train is not None and not df_train.empty:
                st.dataframe(
                    df_train[['id', 'original_title', 'label_text', 'label_num', 'source_platform']],
                    column_config={
                        "id": "ID",
                        "original_title": "Judul Klaim",
                        "label_text": "Label",
                        "label_num": "Biner",
                        "source_platform": "Sumber"
                    },
                    hide_index=True,
                    use_container_width=True
                )
                train_csv_bytes = df_train.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Unduh train_dataset.csv", train_csv_bytes, "train_dataset.csv", "text/csv")
            else:
                st.info("Train dataset belum terbuat.")

        with col_te:
            st.markdown(f"#### 2. Data Uji / Testing Set 20% ({len(df_test) if df_test is not None else 0} Sampel)")
            if df_test is not None and not df_test.empty:
                st.dataframe(
                    df_test[['id', 'original_title', 'label_text', 'label_num', 'source_platform']],
                    column_config={
                        "id": "ID",
                        "original_title": "Judul Klaim",
                        "label_text": "Label",
                        "label_num": "Biner",
                        "source_platform": "Sumber"
                    },
                    hide_index=True,
                    use_container_width=True
                )
                test_csv_bytes = df_test.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Unduh test_dataset.csv", test_csv_bytes, "test_dataset.csv", "text/csv")
            else:
                st.info("Test dataset belum terbuat.")

    # ----------------------------------------------------------
    # SUB-TAB 4: BENCHMARK & EVALUASI MODEL
    # ----------------------------------------------------------
    with subtab_s4:
        st.markdown("### 🤖 Tahap 4: Benchmark & Evaluasi Performa Model")
        st.caption("Perbandingan metrik evaluasi (Akurasi, Presisi, Recall, F1-Score) untuk seluruh model deteksi: NLP Logistic Regression, LSTM Deep Learning, OpenRouter LLM, Cross-Checker, dan Ensemble.")

        nlp_eval = st.session_state.nlp_model.evaluation_metrics
        lstm_eval = st.session_state.lstm_model.evaluation_metrics

        # Matriks Perbandingan Model
        st.markdown("#### Tabel Komparasi Benchmark Model:")
        benchmark_df = get_model_benchmark_table(nlp_eval, lstm_eval)
        st.dataframe(
            benchmark_df,
            column_config={
                "Model / Pendekatan": "Model / Pendekatan",
                "Tipe / Teknologi": "Teknologi & Fitur",
                "Akurasi (%)": "Akurasi",
                "Presisi (%)": "Presisi",
                "Recall (%)": "Recall",
                "F1-Score (%)": "F1-Score",
                "Keunggulan": "Karakteristik Utama"
            },
            hide_index=True,
            use_container_width=True
        )

        st.markdown("---")
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("#### 1. Evaluasi NLP (TF-IDF + Logistic Regression)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Akurasi", f"{nlp_eval['accuracy']}%")
            m2.metric("Presisi", f"{nlp_eval['precision']}%")
            m3.metric("Recall", f"{nlp_eval['recall']}%")
            m4.metric("F1-Score", f"{nlp_eval['f1_score']}%")

            st.markdown("##### Heatmap Confusion Matrix NLP (Data Uji):")
            cm_nlp = nlp_eval.get("confusion_matrix", [[1, 0], [0, 1]])
            fig_nlp = plot_confusion_matrix_heatmap(cm_nlp, model_name="NLP Logistic Regression", cmap="Blues")
            st.pyplot(fig_nlp, use_container_width=False)

        with col_m2:
            st.markdown("#### 2. Evaluasi LSTM (Deep Learning Sequential)")
            l1, l2, l3, l4 = st.columns(4)
            l1.metric("Akurasi", f"{lstm_eval['accuracy']}%")
            l2.metric("Presisi", f"{lstm_eval['precision']}%")
            l3.metric("Recall", f"{lstm_eval['recall']}%")
            l4.metric("F1-Score", f"{lstm_eval['f1_score']}%")

            st.markdown("##### Heatmap Confusion Matrix LSTM (Data Uji):")
            cm_lstm = lstm_eval.get("confusion_matrix", [[1, 0], [0, 1]])
            fig_lstm = plot_confusion_matrix_heatmap(cm_lstm, model_name="LSTM Deep Learning", cmap="PuBuGn")
            st.pyplot(fig_lstm, use_container_width=False)

# ==========================================================
# TAB 3: BASIS DATA RUJUKAN CEK FAKTA
# ==========================================================
with tab_database:
    st.markdown("### Korpus Basis Data Cek Fakta Terverifikasi")
    st.caption("Data rujukan resmi yang digunakan untuk verifikasi silang (Cross-Checker) dan pelatihan model VERITAS-ID.")

    col_f1, col_f2 = st.columns([1.5, 1])
    with col_f1:
        filter_label = st.multiselect("Filter Kelas Label", options=["HOAX", "FAKTA"], default=["HOAX", "FAKTA"])
    with col_f2:
        sort_choice = st.selectbox("Urutkan Berdasarkan Waktu/Tanggal:", options=[
            "Terbaru (Newest First)",
            "Terlama (Oldest First)",
            "ID (Urutan Masuk)"
        ])

    order_mapping = {
        "Terbaru (Newest First)": "created_at DESC",
        "Terlama (Oldest First)": "created_at ASC",
        "ID (Urutan Masuk)": "id ASC"
    }
    selected_order = order_mapping.get(sort_choice, "created_at DESC")

    fact_records = fetch_all_fact_checks(order_by=selected_order)

    if fact_records:
        df = pd.DataFrame(fact_records)
        if "published_at" not in df.columns:
            df["published_at"] = df["created_at"]
        else:
            df["published_at"] = df["published_at"].fillna(df["created_at"])

        df_display = df[['id', 'label', 'created_at', 'source_name', 'category', 'title', 'verdict_details', 'source_url']]
        df_filtered = df_display[df_display['label'].isin(filter_label)]

        st.dataframe(
            df_filtered,
            column_config={
                "id": st.column_config.NumberColumn("ID", format="%d"),
                "label": "Kelas Label",
                "created_at": st.column_config.TextColumn("Tanggal & Waktu Rilis / Input"),
                "source_name": "Lembaga Sumber",
                "category": "Kategori",
                "title": "Judul Artikel / Klaim",
                "verdict_details": "Rincian Verifikasi Fakta",
                "source_url": st.column_config.LinkColumn("Tautan Asli Sumber")
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("Belum ada data rujukan dalam database.")

# ==========================================================
# TAB 4: ADMIN PANEL & PIPELINE RUNNER
# ==========================================================
with tab_admin:
    st.markdown("### Admin Panel - Kontrol Pipeline Runtut (1 -> 2 -> 3 -> 4)")
    st.caption("Picu otomatis pengumpulan artikel berita & hoax terbaru dari TurnBackHoax.id & CekFakta Tempo, jalankan preprocessing, labelling, dan latih ulang model NLP & LSTM.")

    col_adm1, col_adm2 = st.columns([2, 1])

    with col_adm1:
        st.markdown("#### Eksekusi Pipeline Data Lengkap (Tahap 1 - 4)")
        st.write("Menjalankan siklus data mining end-to-end: Scraping -> Preprocessing -> Labelling & Split -> Retraining Model.")

        page_depth = st.slider("Target Kedalaman Paginasi TurnBackHoax (Halaman):", min_value=10, max_value=120, value=70, step=10, help="Menentukan berapa banyak halaman arsip TurnBackHoax yang dijelajahi secara paralel.")
        workers_val = st.slider("Jumlah Multi-Worker Thread Paralel:", min_value=4, max_value=20, value=12, step=2, help="Jumlah thread paralel untuk mempercepat proses crawling data.")

        if st.button("Jalankan Pipeline Lengkap (Tahap 1 -> 4) Sekarang", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Tahap 1
            status_text.write(" **Tahap 1/4:** Menjalankan Multi-Threaded Web Scraping (TurnBackHoax & Tempo CekFakta 2026)...")
            progress_bar.progress(25)
            scrape_res = run_scraping_pipeline(max_pages=page_depth, max_workers=workers_val)

            # Tahap 2
            status_text.write(" **Tahap 2/4:** Menjalankan Preprocessing Teks (Case Folding, Cleaning, Stopwords, Tokenisasi)...")
            progress_bar.progress(50)
            prep_res = run_preprocessing_pipeline()

            # Tahap 3
            status_text.write(" **Tahap 3/4:** Menjalankan Binary Labelling & Stratified Train/Test Split (80:20)...")
            progress_bar.progress(75)
            label_res = run_labelling_pipeline()

            # Tahap 4
            status_text.write(" **Tahap 4/4:** Melatih Ulang Model NLP & LSTM serta Membangun Matriks Evaluasi...")
            progress_bar.progress(90)
            st.session_state.nlp_model.train_model()
            st.session_state.lstm_model.train_model()
            st.session_state.cross_checker.load_and_build_index()

            progress_bar.progress(100)
            status_text.empty()

            st.success(f"✅ Pipeline Sukses! Mengumpulkan {scrape_res.get('total_scraped', 0)} data seimbang ({scrape_res.get('fakta_percentage')} FAKTA : {scrape_res.get('hoax_percentage')} HOAX). Seluruh dataset dan model telah disinkronkan!")

            st.json({
                "tahap_1_scraping": scrape_res,
                "tahap_2_preprocessing": prep_res,
                "tahap_3_labelling": label_res,
                "tahap_4_nlp_metrics": st.session_state.nlp_model.evaluation_metrics,
                "tahap_4_lstm_metrics": st.session_state.lstm_model.evaluation_metrics
            })

    with col_adm2:
        st.markdown("#### Status Korpus Saat Ini")
        fact_records = fetch_all_fact_checks()
        hoax_cnt = len([r for r in fact_records if r['label'] == 'HOAX'])
        fakta_cnt = len([r for r in fact_records if r['label'] == 'FAKTA'])

        st.metric("Total Data Cek Fakta", len(fact_records))
        st.metric("Jumlah Data HOAX", hoax_cnt)
        st.metric("Jumlah Data FAKTA", fakta_cnt)

        st.markdown("---")
        if st.button("🧹 Reset & Muat Ulang Data Awal (TurnBackHoax & Tempo)", use_container_width=True):
            if os.path.exists(DB_PATH):
                os.remove(DB_PATH)
            init_database(DB_PATH)
            run_preprocessing_pipeline()
            run_labelling_pipeline()
            st.session_state.nlp_model.train_model()
            st.session_state.lstm_model.train_model()
            st.session_state.cross_checker.load_and_build_index()
            st.success("✅ Database & dataset berhasil di-reset!")
            st.rerun()

    st.markdown("---")
    st.markdown("### Pengaturan API OpenRouter LLM")
    st.caption("Konfigurasikan OpenRouter API Key agar sistem dapat mengekstrak klaim utama dan merangkum berita secara otomatis menggunakan AI.")

    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "data", "config.json")
    saved_cfg = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved_cfg = json.load(f)
        except:
            pass

    col_llm1, col_llm2 = st.columns([1.5, 1])
    with col_llm1:
        existing_key = saved_cfg.get("openrouter_api_key", saved_cfg.get("llm_api_key", ""))
        existing_model = saved_cfg.get("openrouter_model", saved_cfg.get("llm_model", DEFAULT_OPENROUTER_MODEL))

        cfg_llm_key = st.text_input("OpenRouter API Key:", value=existing_key, type="password", placeholder="sk-or-v1-...")
        cfg_llm_model = st.text_input("Nama Model OpenRouter:", value=existing_model, placeholder=DEFAULT_OPENROUTER_MODEL)

        btn_save_or, btn_test_or = st.columns(2)
        with btn_save_or:
            if st.button("💾 Simpan Pengaturan OpenRouter", use_container_width=True):
                saved_cfg["openrouter_api_key"] = cfg_llm_key.strip()
                saved_cfg["llm_api_key"] = cfg_llm_key.strip()
                saved_cfg["openrouter_model"] = cfg_llm_model.strip() if cfg_llm_model.strip() else DEFAULT_OPENROUTER_MODEL
                saved_cfg["llm_model"] = cfg_llm_model.strip() if cfg_llm_model.strip() else DEFAULT_OPENROUTER_MODEL
                os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
                with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                    json.dump(saved_cfg, f, indent=4)
                st.success("✅ Pengaturan OpenRouter API Key berhasil disimpan!")

        with btn_test_or:
            if st.button("Uji Koneksi OpenRouter API", type="primary", use_container_width=True):
                if not cfg_llm_key.strip():
                    st.error("⚠️ Masukkan OpenRouter API Key terlebih dahulu.")
                else:
                    with st.spinner("Menguji koneksi ke OpenRouter API..."):
                        test_or_res = test_openrouter_connection(cfg_llm_key.strip(), cfg_llm_model.strip())
                        if test_or_res["success"]:
                            st.success(f"✅ {test_or_res['message']}")
                        else:
                            st.error(f"❌ {test_or_res['message']}")

    with col_llm2:
        st.markdown("#### 💡 Cara Mendapatkan API Key OpenRouter")
        st.markdown("""
            1. Kunjungi [OpenRouter.ai Keys](https://openrouter.ai/keys).
            2. Sign In dengan akun Google / GitHub Anda.
            3. Klik tombol **Create Key** dan salin kunci yang diawali `sk-or-v1-...`.
            4. Tempelkan pada formulir di sebelah kiri.
            5. Model gratis rekomendasi: `google/gemini-2.0-flash-exp:free` atau `meta-llama/llama-3.3-70b-instruct:free`.
        """)

# ==========================================================
# TAB 5: TENTANG & PRD
# ==========================================================
with tab_about:
    st.markdown("""
        ### PRD & Arsitektur VERITAS-ID
        **Sistem Deteksi Hoax Multi-Sumber Berbasis NLP, Deep Learning & LLM**
        
        #### 5 Tahapan Alur Data:
        1. **Tahap 1: Scraping Mentah** -> Mengambil artikel hoax dan klarifikasi dari TurnBackHoax.id & CekFakta Tempo, disimpan ke `data/raw/raw_scraped_dataset.csv`.
        2. **Tahap 2: Preprocessing Teks** -> Case folding, pembersihan URL/mention/simbol, normalisasi rupiah/persen, stopwords removal Bahasa Indonesia, dan tokenisasi ke `data/processed/preprocessed_dataset.csv`.
        3. **Tahap 3: Labelling & Split** -> Pelabelan biner (`HOAX` = 1, `FAKTA` = 0) dan pembagian Stratified 80:20 Train/Test ke `data/labeled/`.
        4. **Tahap 4: Modelling & Evaluasi** -> Pelatihan NLP Logistic Regression, LSTM Deep Learning, OpenRouter LLM, Cross-Checker, dan evaluasi metrik (Akurasi, Presisi, Recall, F1, Confusion Matrix).
        5. **Tahap 5: Dashboard Visual** -> Penyajian hasil tahapan 1 sampai 4 secara interaktif pada Tab *Pipeline & Evaluasi Dataset*.
    """)

# ==========================================================
# SIDEBAR: RIWAYAT PENGECEKAN
# ==========================================================
with st.sidebar:
    st.markdown("### Riwayat Pengecekan")
    history_items = get_recent_history(limit=8)
    if history_items:
        for item in history_items:
            badge_icon = "🔴" if "HOAX" in item["predicted_label"] else ("🟢" if "FAKTA" in item["predicted_label"] else "🟠")
            st.markdown(f"""
                **{badge_icon} {item['predicted_label']}** ({item['confidence_score']}%)  
                <small><em>"{item['input_text'][:45]}..."</em></small>  
                <small style="color: grey;">{item['checked_at']}</small>
                ---
            """, unsafe_allow_html=True)
    else:
        st.caption("Belum ada riwayat pengecekan dalam sesi ini.")
