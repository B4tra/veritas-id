from modules.llm_extractor import test_openrouter_connection
import os
import streamlit as st
import pandas as pd

# Initialize database seed if needed
from modules.seed_data import init_database, DB_PATH
init_database(DB_PATH)

from modules.database import save_search_history, get_recent_history, fetch_all_fact_checks
from modules.url_parser import is_valid_url, extract_content_from_url
from modules.cross_checker import CrossChecker
from modules.nlp_engine import HoaxDetectorModel
from modules.lstm_engine import LSTMDetectionModel
from modules.scraper_engine import run_scraping_pipeline
from modules.email_service import send_gmail_report, send_test_email
from modules.llm_extractor import extract_claim_with_llm, test_llm_connection, get_llm_verdict, DEFAULT_GEMINI_MODEL, DEFAULT_OPENROUTER_MODEL
from modules.ensemble_engine import compute_ensemble_verdict
import json

# Page Configuration
st.set_page_config(
    page_title="VERITAS-ID | Sistem Deteksi Hoax NLP Indonesia",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "nlp_model" not in st.session_state:
    st.session_state.nlp_model = HoaxDetectorModel()
if "lstm_model" not in st.session_state:
    st.session_state.lstm_model = LSTMDetectionModel()
if "cross_checker" not in st.session_state:
    st.session_state.cross_checker = CrossChecker()

# Header Section
st.markdown("""
    <div class="main-header">
        <h1>🛡️ VERITAS-ID</h1>
        <p>Sistem Deteksi Hoax Berbasis NLP dan Korpus Multi-Sumber Cek Fakta Indonesia<br>
        <small style="color: #64748b;">(TurnBackHoax.id / MAFINDO • Tempo CekFakta • CNN Indonesia)</small></p>
    </div>
""", unsafe_allow_html=True)

# Navigation Tabs
tab_detect, tab_database, tab_admin, tab_about = st.tabs(["Pengecekan Klaim", "Basis Data Rujukan Cek Fakta", "Admin Panel Retrain", "Tentang & PRD"])

# TAB 1: PENGECEKAN KLAIM (MAIN CORE FEATURE)
with tab_detect:
    col_input, col_sidebar_info = st.columns([2, 1])

    with col_input:
        st.markdown("### Input Teks Klaim atau Tautan Berita")
        st.caption("Masukkan naskah klaim, pesan berantai WhatsApp, atau tempelkan URL tautan artikel berita untuk dianalisis.")
        
        user_input = st.text_area(
            label="Kotak Input Klaim/Tautan",
            value=st.session_state.get("current_input", ""),
            placeholder="Tempel teks klaim atau tautan berita di sini (Contoh: https://... atau 'Makan telur rebus jam 12 malam bisa mencegah Corona...')",
            height=140,
            label_visibility="collapsed"
        )
        
        # Sync user typing to session state
        if user_input != st.session_state.get("current_input", ""):
            st.session_state.current_input = user_input

        col_btn1, col_btn2, _ = st.columns([2, 1.2, 2])
        with col_btn1:
            if st.button("Langkah 1: Ekstrak Klaim (OpenRouter LLM)", type="primary", use_container_width=True):
                if not user_input.strip():
                    st.warning("Mohon masukkan teks klaim atau tautan berita terlebih dahulu.")
                else:
                    input_text = user_input.strip()
                    is_url_input = is_valid_url(input_text)
                    raw_text_for_llm = input_text

                    if is_url_input:
                        st.toast("Tautan berita terdeteksi! Mengekstraksi konten artikel...", icon="ℹ️")
                        url_res = extract_content_from_url(input_text)
                        if url_res["success"]:
                            raw_text_for_llm = url_res["text"]

                    # Load openrouter config
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

                    with st.spinner("Menjalankan ekstraksi klaim abstrak via OpenRouter AI..."):
                        llm_res = extract_claim_with_llm(raw_text_for_llm, api_key=api_key, model_name=model_name)
                        st.session_state.llm_result = llm_res
                        st.session_state.edited_claim = llm_res["extracted_claim"]
                        st.session_state.show_extraction_card = True
                        st.session_state.show_results = False

        with col_btn2:
            if st.button("💡 Sampel Hoax", use_container_width=True):
                st.session_state.current_input = "Beredar pesan berantai di WhatsApp yang mengklaim bahwa memakan telur rebus pada jam 12 malam secara ajaib dapat menangkal dan menyembuhkan infeksi virus Corona."
                st.session_state.show_extraction_card = False
                st.session_state.show_results = False
                st.rerun()

    with col_sidebar_info:
        st.markdown("### Ringkasan Sesi & Statistik")
        
        recent = get_recent_history(limit=5)
        st.metric(label="Total Pengecekan Riwayat", value=len(recent))

    # DISPLAY STEP 1: LLM CLAIM EXTRACTION REVIEW CARD
    if st.session_state.get("show_extraction_card", False) and "llm_result" in st.session_state:
        res = st.session_state.llm_result
        st.markdown("---")
        st.markdown("### Hasil Ekstraksi Klaim Inti AI & Entitas")

        if res.get("api_key_missing", False):
            st.warning(f"""
                {res['message']}  
                👉 **Panduan:** Masukkan API Key OpenRouter Anda (`sk-or-v1-...`) pada **Tab ⚙️ Admin Panel Retrain** -> bagian **Pengaturan API OpenRouter LLM**.
            """)
        elif not res.get("is_llm", False):
            st.info(f"{res['message']}")
        else:
            st.success(f"{res['message']} (Provider: `OpenRouter`, Model: `{res.get('model_used', DEFAULT_OPENROUTER_MODEL)}`)")

        col_card1, col_card2 = st.columns([1.8, 1.2])
        with col_card1:
            st.markdown("#### Tinjau & Edit Kalimat Klaim Utama:")
            edited_text = st.text_area(
                label="Kalimat Klaim Utama (Dapat disunting):",
                value=st.session_state.get("edited_claim", res["extracted_claim"]),
                height=90,
                key="edited_claim_input",
                help="Kalimat klaim ini yang akan dikirimkan ke model klasifikasi NLP & basis data rujukan cek fakta."
            )
            st.session_state.edited_claim = edited_text

        with col_card2:
            st.markdown("#### Informasi Tambahan Artikel:")
            st.write(f"**Ringkasan Teks:** {res['summary']}")
            if res.get("key_entities"):
                tags_html = " ".join([f'<span class="word-tag">{e}</span>' for e in res["key_entities"]])
                st.markdown(f"**Entitas Terdeteksi:**<br>{tags_html}", unsafe_allow_html=True)
            st.caption(f"Tingkat Provokatif/Sensasional: **{res.get('sensational_rating', 'RENDAH')}**")

        st.write("")
        if st.button("Langkah 2: Periksa & Verifikasi Fakta Sekarang", type="primary", use_container_width=True):
            st.session_state.show_results = True

    # STEP 2: RUN VERIFICATION IF TRIGGERED
    if st.session_state.get("show_results", False):
        analysis_text = st.session_state.get("edited_claim", st.session_state.get("current_input", "")).strip()
        input_text = st.session_state.get("current_input", "").strip()
        is_url_input = is_valid_url(input_text)
        
        with st.spinner("Memproses 4 Sinyal Deteksi (NLP + LSTM + OpenRouter LLM + Cross-Checker)..."):
            # 1. NLP Model Classification
            prediction = st.session_state.nlp_model.predict(analysis_text)
            
            # 2. LSTM Model Classification
            prediction_lstm = st.session_state.lstm_model.predict(analysis_text)
            
            # 3. OpenRouter LLM Zero-shot Verdict (5 Categories)
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

            # 4. Cross-Verification Search
            matches = st.session_state.cross_checker.find_matches(analysis_text)
            top_match = matches[0] if matches else None

            # 5. Compute Integrated Ensemble Verdict
            ensemble_res = compute_ensemble_verdict(prediction, prediction_lstm, llm_verdict_res, matches)

            # Save to Database Search History
            save_search_history(
                input_text=analysis_text,
                input_type="url" if is_url_input else "text",
                predicted_label=ensemble_res["final_label"],
                confidence_score=ensemble_res["confidence_score"],
                matched_reference_id=top_match["id"] if top_match else None
            )

        st.markdown("---")
        st.markdown("## Hasil Verdict Ensemble Multi-Model")
        
        # --- ENSEMBLE SUMMARY CARD ---
        ens_badge_class = f"badge-{ensemble_res['badge_type']}"
        
        # Color & icon mapping for 5 LLM verdict categories
        verdict_badge_colors = {
            "BENAR": "#10b981",         # Green
            "SEBAGIAN BENAR": "#06b6d4", # Cyan
            "BELUM ADA BUKTI": "#64748b",# Slate Gray
            "SESAT": "#f59e0b",          # Amber/Orange
            "KELIRU": "#ef4444"          # Red
        }
        v_color = verdict_badge_colors.get(ensemble_res["llm_verdict"], "#64748b")
        
        col_ens1, col_ens2 = st.columns([1.3, 1.7])
        with col_ens1:
            st.markdown("#### Verdict Akhir Ensemble")
            st.markdown(f'<div class="badge-result {ens_badge_class}">{ensemble_res["final_label"]}</div>', unsafe_allow_html=True)
            st.write("")
            st.markdown(f"**Ringkasan:** {ensemble_res['verdict_summary']}")
            
            # Show 5-Category LLM Verdict Badge
            st.markdown(f"""
                <div style="margin-top: 1rem; padding: 0.8rem; border-radius: 8px; background: rgba(0,0,0,0.04); border-left: 4px solid {v_color};">
                    <small style="color: #64748b; font-weight: bold;">VONIS CEK FAKTA LLM:</small><br>
                    <span style="font-size: 1.1rem; font-weight: bold; color: {v_color};">🏷️ {ensemble_res['llm_verdict']}</span><br>
                    <small style="color: #334155;"><em>"{ensemble_res['llm_reasoning']}"</em></small>
                </div>
            """, unsafe_allow_html=True)

        with col_ens2:
            st.markdown("#### Indikator Risiko Hoaks (Ensemble Score)")
            hoax_pct = ensemble_res["hoax_score_percent"]
            st.progress(hoax_pct / 100.0)
            
            em1, em2, em3 = st.columns(3)
            em1.metric("Skor Risiko Hoaks", f"{hoax_pct}%")
            em2.metric("Tingkat Keandalan Fakta", f"{ensemble_res['fakta_score_percent']}%")
            em3.metric("Status Vonis LLM", ensemble_res["llm_verdict"])

            st.markdown("##### Matriks Kontribusi Sinyal Ensemble:")
            breakdown = ensemble_res["breakdown"]
            st.markdown(f"""
            | Sinyal Deteksi | Bobot | Hasil Prediksi Sinyal | Skor Hoaks |
            | :--- | :---: | :--- | :---: |
            | **NLP (Logistic Reg. + Heuristics)** | 25% | `{breakdown['nlp']['label']}` | **{breakdown['nlp']['hoax_prob']}%** |
            | **LSTM (Deep Learning)** | 20% | `{breakdown['lstm']['label']}` | **{breakdown['lstm']['hoax_prob']}%** |
            | **OpenRouter LLM (Zero-shot)** | 35% | ` {breakdown['llm']['verdict']}` | **{breakdown['llm']['hoax_prob']}%** |
            | **Cross-Checker (Corpus Match)** | 20% | `{breakdown['cross_checker']['top_match'][:35]}...` | **{breakdown['cross_checker']['hoax_prob']}%** |
            """)

        # EXPLAINABILITY KEYWORDS
        if prediction.get("suspicious_words"):
            st.markdown("#####  Kata/Frasa Mencurigakan Terdeteksi (Explainability):")
            tags_html = "".join([f'<span class="word-tag"> {word}</span>' for word in prediction["suspicious_words"]])
            st.markdown(f'<div style="margin-bottom: 1rem;">{tags_html}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # BREAKDOWN DETAIL TABS
        st.markdown("### Rincian Analisis per Sinyal")
        tab_logreg, tab_lstm, tab_llm_detail = st.tabs(["Logistic Regression (Klasik)", "LSTM (Deep Learning)", "OpenRouter LLM Analysis"])

        # PART (A1): LOGISTIC REGRESSION RESULTS
        with tab_logreg:
            res_col1, res_col2 = st.columns([1.2, 1.8])
            with res_col1:
                st.markdown("#### Status Prediksi NLP")
                badge_class = f"badge-{prediction['badge_type']}"
                st.markdown(f'<div class="badge-result {badge_class}">{prediction["label"]}</div>', unsafe_allow_html=True)
                st.write("")
                st.caption(prediction["note"])

            with res_col2:
                st.markdown("#### Skor Keyakinan Model")
                score = prediction["confidence_score"]
                st.progress(score / 100.0)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Tingkat Keyakinan", f"{score}%")
                m2.metric("Probabilitas Hoax", f"{prediction['hoax_prob']}%")
                m3.metric("Probabilitas Fakta", f"{prediction['fakta_prob']}%")
                
        # PART (A2): LSTM RESULTS
        with tab_lstm:
            res_col3, res_col4 = st.columns([1.2, 1.8])
            with res_col3:
                st.markdown("#### Status Prediksi LSTM")
                badge_class_lstm = f"badge-{prediction_lstm['badge_type']}"
                st.markdown(f'<div class="badge-result {badge_class_lstm}">{prediction_lstm["label"]}</div>', unsafe_allow_html=True)
                st.write("")
                st.caption(prediction_lstm["note"])

            with res_col4:
                st.markdown("#### Skor Keyakinan LSTM")
                score_lstm = prediction_lstm["confidence_score"]
                st.progress(score_lstm / 100.0)
                
                m4, m5, m6 = st.columns(3)
                m4.metric("Tingkat Keyakinan", f"{score_lstm}%")
                m5.metric("Probabilitas Hoax", f"{prediction_lstm['hoax_prob']}%")
                m6.metric("Probabilitas Fakta", f"{prediction_lstm['fakta_prob']}%")

        # PART (A3): LLM DETAIL
        with tab_llm_detail:
            st.markdown(f"#### Vonis Analisis LLM (`{llm_verdict_res.get('model_used', model_name)}`)")
            st.info(f"**Vonis Kategori:** {llm_verdict_res['verdict']}\n\n**Keyakinan Model:** {llm_verdict_res['confidence']}%\n\n**Penalaran:** {llm_verdict_res['reasoning']}")

        st.markdown("---")

        # PART (B): CROSS-VERIFICATION REFERENCES
        st.markdown("##Rujukan Verifikasi Silang Data Cek Fakta")
        
        if matches:
            st.success(f"Ditemukan {len(matches)} rujukan artikel cek fakta serupa dari basis data terverifikasi:")
            for m in matches:
                ref_type = "hoax" if m["label"] == "HOAX" else "fakta"
                st.markdown(f"""
                    <div class="ref-card {ref_type}">
                        <h4><a href="{m['source_url']}" target="_blank" style="text-decoration: none; color: inherit;">🔗 {m['title']}</a></h4>
                        <p><strong>Sumber Resmi:</strong> {m['source_name']} | <strong>Kategori:</strong> {m['category']} | <strong>Kemiripan Teks:</strong> {m['similarity_score']}%</p>
                        <p style="background: rgba(0,0,0,0.1); padding: 0.8rem; border-radius: 6px;"><strong>📌 Vonis & Bukti Pendukung:</strong><br>{m['verdict_details']}</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Tidak ditemukan klaim yang persis serupa di basis data rujukan (TurnBackHoax/Tempo/CNN). Hasil prediksi murni didasarkan pada analisis model klasifikasi NLP & LLM.")

        # PART (C): DISCLAIMER / CATATAN PENGINGAT
        st.markdown("""
            <div class="disclaimer-box">
                <strong>⚠️ Catatan Pengingat & Etika Penggunaan:</strong><br>
                Hasil deteksi ini merupakan alat bantu triase awal berbasis AI/NLP dan pencocokan data historis, <strong>bukan vonis mutlak final</strong>. Selalu lakukan verifikasi mandiri dan rujuk lembaga cek fakta independen (MAFINDO / Tempo / Kominfo) sebelum menyebarkan informasi.
            </div>
        """, unsafe_allow_html=True)



# TAB 2: BASIS DATA RUJUKAN CEK FAKTA
with tab_database:
    st.markdown("###Korpus Multi-Sumber Cek Fakta Indonesia")
    st.caption("Basis data rujukan terstruktur yang digunakan untuk verifikasi silang dan pelatihan model NLP VERITAS-ID.")
    
    col_f1, col_f2 = st.columns([1.5, 1])
    with col_f1:
        filter_label = st.multiselect("Filter Kelas Label", options=["HOAX", "FAKTA"], default=["HOAX", "FAKTA"])
    with col_f2:
        sort_choice = st.selectbox("⏳ Urutkan Berdasarkan Waktu/Tanggal:", options=[
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

    try:
        fact_records = fetch_all_fact_checks(order_by=selected_order)
    except TypeError:
        import importlib
        import modules.database
        importlib.reload(modules.database)
        from modules.database import fetch_all_fact_checks
        fact_records = fetch_all_fact_checks(order_by=selected_order)
    if fact_records:
        df = pd.DataFrame(fact_records)
        
        # Ensure published_at / created_at is present
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
                "created_at": st.column_config.TextColumn("📅 Tanggal & Waktu Rilis / Input"),
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

# TAB 3: ADMIN PANEL (SCRAPING & RETRAIN)
with tab_admin:
    st.markdown("### ⚙️ Admin Panel - Pipeline Web Scraper & Retraining")
    st.caption("Picu otomatis pengumpulan artikel berita & hoax terbaru dari TurnBackHoax.id, CNN Indonesia, dan LKBN ANTARA, lalu latih ulang model NLP & LSTM.")
    
    col_adm1, col_adm2 = st.columns([2, 1])
    with col_adm1:
        st.markdown("#### Kontrol Retraining Pipeline")
        st.write("Proses ini akan mengunduh artikel berita/hoax terbaru, memperbarui database SQLite, dan melatih ulang model secara otomatis.")
        
        limit_val = st.slider("Jumlah Artikel per Sumber untuk Diambil:", min_value=1, max_value=10, value=3)
        
        if st.button("🚀 Jalankan Web Scraper & Retrain Model Sekarang", type="primary", use_container_width=True):
            with st.spinner("⏳ Menjalankan scraper & melatih ulang model NLP + LSTM (Mohon tunggu beberapa saat)..."):
                scrape_res = run_scraping_pipeline(limit_per_source=limit_val)
                
                # Re-train models in session state
                st.session_state.nlp_model.train_model()
                st.session_state.lstm_model.train_model()
                st.session_state.cross_checker.load_and_build_index()
                
                st.success(f"✅ Pipeline Berhasil! Menambahkan {scrape_res['inserted_count']} artikel baru ke database. Model NLP & LSTM telah diperbarui!")
                st.json(scrape_res)
                
    with col_adm2:
        st.markdown("####Status Database Saat Ini")
        fact_records = fetch_all_fact_checks()
        hoax_cnt = len([r for r in fact_records if r['label'] == 'HOAX'])
        fakta_cnt = len([r for r in fact_records if r['label'] == 'FAKTA'])
        
        st.metric("Total Data Cek Fakta", len(fact_records))
        st.metric("Jumlah Data HOAX", hoax_cnt)
        st.metric("Jumlah Data FAKTA", fakta_cnt)
        
        st.markdown("---")
        if st.button("🧹 Kosongkan Database (0 Data)", use_container_width=True):
            if os.path.exists(DB_PATH):
                os.remove(DB_PATH)
            init_database(DB_PATH)
            st.session_state.nlp_model.train_model()
            st.session_state.lstm_model.train_model()
            st.session_state.cross_checker.load_and_build_index()
            st.success("✅ Database berhasil dikosongkan (0 Data)!")
            st.rerun()

    st.markdown("---")
    st.markdown("### 🤖 Pengaturan API OpenRouter LLM")
    st.caption("Konfigurasikan OpenRouter API Key agar sistem dapat mengekstrak klaim utama dan merangkum berita secara otomatis menggunakan AI.")

    # Load existing config if available
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

        cfg_llm_key = st.text_input(
            "OpenRouter API Key:",
            value=existing_key,
            type="password",
            placeholder="sk-or-v1-...",
            help="Dapatkan API Key gratis di https://openrouter.ai/keys"
        )
        cfg_llm_model = st.text_input(
            "Nama Model OpenRouter:",
            value=existing_model,
            placeholder="google/gemini-2.0-flash-exp:free"
        )

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
                st.success("✅ Pengaturan OpenRouter API Key berhasil disimpan ke config.json!")

        with btn_test_or:
            if st.button("🧪 Uji Koneksi OpenRouter API", type="primary", use_container_width=True):
                if not cfg_llm_key.strip():
                    st.error("⚠️ API Key OpenRouter belum diisi. Masukkan API Key terlebih dahulu.")
                else:
                    with st.spinner("Menghubungkan ke OpenRouter API..."):
                        test_or_res = test_openrouter_connection(cfg_llm_key.strip(), cfg_llm_model.strip())
                        if test_or_res["success"]:
                            st.success(f"✅ {test_or_res['message']}")
                        else:
                            st.error(f"❌ {test_or_res['message']}")

    with col_llm2:
        st.markdown("#### 💡 Cara Mendapatkan API Key OpenRouter Gratis")
        st.markdown("""
            1. Buka situs [OpenRouter.ai Keys](https://openrouter.ai/keys).
            2. Buat akun gratis / Sign In dengan Google/GitHub.
            3. Klik tombol **Create Key** -> Beri nama key (misal: `VERITAS-ID`).
            4. Salin kunci rahasia yang diawali dengan `sk-or-v1-...`.
            5. Tempelkan pada kolom **OpenRouter API Key** di samping kiri.
            6. Model gratis default: `google/gemini-2.0-flash-exp:free` atau `meta-llama/llama-3.3-70b-instruct:free`.
        """)

    st.markdown("---")
    st.markdown("###Pengaturan Notifikasi Email Gmail (Penjadwalan Harian)")
    st.caption("Konfigurasikan Gmail agar sistem mengirimkan laporan ringkasan berita & hoax otomatis setiap hari ke inbox Anda.")
    
    # Load existing config if available
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "data", "config.json")
    saved_cfg = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved_cfg = json.load(f)
        except:
            pass

    col_mail1, col_mail2 = st.columns([1.5, 1])
    with col_mail1:
        cfg_sender = st.text_input("Email Gmail Pengirim:", value=saved_cfg.get("sender_email", ""), placeholder="nama_anda@gmail.com")
        cfg_app_pass = st.text_input("Gmail App Password (16 Digit):", value=saved_cfg.get("app_password", ""), type="password", help="Dapatkan Sandi Aplikasi 16-digit dari Akun Google > Keamanan > Verifikasi 2 Langkah > Sandi Aplikasi.")
        cfg_recipient = st.text_input("Email Penerima Laporan:", value=saved_cfg.get("recipient_email", ""), placeholder="penerima_laporan@gmail.com")

        btn_save_cfg, btn_test_cfg = st.columns(2)
        with btn_save_cfg:
            if st.button("💾 Simpan Pengaturan Email", use_container_width=True):
                new_cfg = {
                    "sender_email": cfg_sender.strip(),
                    "app_password": cfg_app_pass.strip(),
                    "recipient_email": cfg_recipient.strip()
                }
                os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
                with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                    json.dump(new_cfg, f, indent=4)
                st.success("Pengaturan email berhasil disimpan ke config.json!")

        with btn_test_cfg:
            if st.button("Uji Coba Kirim Email Sekarang", type="primary", use_container_width=True):
                if not cfg_sender or not cfg_app_pass or not cfg_recipient:
                    st.error("Mohon lengkapi ketiga kolom email & App Password di atas terlebih dahulu.")
                else:
                    with st.spinner("Menghubungkan ke server Gmail SMTP dan mengirimkan email uji coba..."):
                        import importlib
                        import modules.email_service
                        importlib.reload(modules.email_service)
                        test_res = modules.email_service.send_test_email(cfg_sender, cfg_app_pass, cfg_recipient)
                        if test_res["success"]:
                            st.success(f"✅ {test_res['message']}")
                        else:
                            st.error(f"❌ {test_res['message']}")

    with col_mail2:
        st.markdown("#### Cara Mengaktifkan Penjadwalan Otomatis (Windows)")
        st.markdown("""
            Agar skrip berjalan otomatis **setiap pukul 08:00 WIB**:
            1. Buka **Windows Task Scheduler** di PC Anda.
            2. Klik **Create Basic Task** -> Beri nama `VERITAS-ID Daily Crawl`.
            3. Pilih **Trigger: Daily** -> Pukul `08:00:00`.
            4. Pilih **Action: Start a program**:
               - **Program/script**: `python`
               - **Add arguments**: `daily_scheduler.py`
               - **Start in**: `C:\Gemastik Detection Hoax`
        """)

# TAB 3: TENTANG & PRD
with tab_about:
    st.markdown("""
        ### PRD VERITAS-ID
        **Status:** Draft v1.0 (Dikembangkan untuk GEMASTIK XIX 2026 Divisi Penambangan Data)
        
        #### Target Pengguna:
        1. **Masyarakat Umum:** Memeriksa pesan berantai WhatsApp/Media sosial secara mandiri.
        2. **Pendidik / Aktivis Literasi Digital:** Bahan edukasi & demonstrasi verifikasi fakta.
        3. **Jurnalis / Fact-checker Pemula:** Alat bantu triase awal sebelum verifikasi manual mendalam.
        
        #### Sumber Data Pelatihan:
        - **TurnBackHoax.id (MAFINDO):** Kelas HOAX (API Yudistira & Arsip Publik).
        - **Tempo.co (CekFakta & Berita Reguler):** Kelas FAKTA.
        - **CNN Indonesia:** Kelas FAKTA Tambahan.
    """)

# SIDEBAR: RIWAYAT PENCARIAN SESI (P1 Feature)
with st.sidebar:
    st.markdown("###Riwayat Pengecekan Sesi")
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
