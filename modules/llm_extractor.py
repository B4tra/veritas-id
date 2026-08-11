# Modul ini berfungsi untuk integrasi OpenRouter API dalam melakukan ekstraksi klaim berita
# dan fact-checking (cek fakta) menggunakan model AI (LLM).
import os
import re
import json
import requests

DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
DEFAULT_OPENROUTER_MODEL = DEFAULT_MODEL
DEFAULT_GEMINI_MODEL = DEFAULT_MODEL
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

FREE_MODEL_FALLBACKS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-26b-a4b-it:free",
    "openai/gpt-oss-20b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "google/gemma-4-31b-it:free"
]

# Membersihkan dan menormalkan input API key OpenRouter untuk menghindari format yang salah.
def sanitize_api_key(api_key):
    """Clean and normalize OpenRouter API key input."""
    if not api_key:
        return ""
    key = str(api_key).strip()
    key = key.strip('"\'')
    if key.lower().startswith("authorization:"):
        key = key[14:].strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key.strip('"\'')

# Menggunakan ekstraksi klaim secara heuristik menggunakan NLP dan regex 
# sebagai fallback ketika layanan LLM OpenRouter tidak tersedia.
def fallback_heuristic_extraction(text, is_url_text=False):
    """
    Fallback claim sentence extractor using NLP heuristics & regex when OpenRouter LLM is unavailable.
    """
    if not text:
        return {
            "success": False,
            "is_llm": False,
            "extracted_claim": "",
            "key_entities": [],
            "summary": "",
            "sensational_rating": "RENDAH",
            "message": "Teks kosong."
        }

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Extract sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    clean_sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    # Core claim estimation
    if len(clean_sentences) >= 1:
        extracted_claim = clean_sentences[0]
        if len(extracted_claim) < 30 and len(clean_sentences) > 1:
            extracted_claim = f"{clean_sentences[0]} {clean_sentences[1]}"
    else:
        extracted_claim = text[:150]

    # Summary estimation
    summary = " ".join(clean_sentences[:3]) if clean_sentences else text[:300]

    # Extract key entities (figures, dates, currencies)
    entities = []
    date_matches = re.findall(r'\b(?:\d{1,2}\s+(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember|\d{1,2}|\w+)|jam\s+\d{1,2}|malam|pagi|siang)\b', text, re.IGNORECASE)
    num_matches = re.findall(r'\b(?:Rp\s?[\d.,]+|[\d.,]+\s?(?:juta|miliar|triliun|persen|%|orang|ribu))\b', text, re.IGNORECASE)
    
    entities.extend(date_matches[:3])
    entities.extend(num_matches[:3])
    unique_entities = list(set([e.strip() for e in entities if len(e.strip()) > 1]))

    # Sensationalism score heuristic
    sensational_words = ["segeralah", "sebarkanlah", "viral", "awas", "bahaya", "ajaib", "gratis", "hadiah", "rahasia", "porcine"]
    sensational_count = sum(1 for w in sensational_words if w in text.lower())
    rating = "TINGGI" if sensational_count >= 2 else ("SEDANG" if sensational_count == 1 else "RENDAH")

    return {
        "success": True,
        "is_llm": False,
        "extracted_claim": extracted_claim,
        "key_entities": unique_entities,
        "summary": summary,
        "sensational_rating": rating,
        "message": "Menggunakan ekstraksi heuristik lokal (OpenRouter API Key belum dikonfigurasi/offline)."
    }

# Ekstraksi pernyataan klaim utama, entitas kunci, dan ringkasan menggunakan model LLM OpenRouter.
# Terdapat mekanisme fallback model otomatis jika model utama terkena rate limit (HTTP 429).
def extract_claim_with_llm(raw_text, api_key=None, model_name=DEFAULT_MODEL, provider="openrouter"):
    """
    Extract single core claim statement, key entities, and summary using OpenRouter LLM.
    Includes automatic model fallback if rate-limited (HTTP 429).
    """
    clean_input = raw_text.strip()
    if not clean_input:
        return {
            "success": False,
            "is_llm": False,
            "extracted_claim": "",
            "key_entities": [],
            "summary": "",
            "sensational_rating": "RENDAH",
            "message": "Input teks kosong."
        }

    clean_key = sanitize_api_key(api_key)

    if not clean_key:
        res = fallback_heuristic_extraction(clean_input)
        res["api_key_missing"] = True
        res["message"] = "⚠️ API Key OpenRouter belum diatur. Menampilkan ekstraksi heuristik lokal. Silakan atur OpenRouter API Key di Admin Panel."
        return res

    preferred_model = model_name.strip() if model_name and model_name.strip() else DEFAULT_MODEL
    candidate_models = [preferred_model] + [m for m in FREE_MODEL_FALLBACKS if m != preferred_model]

    headers = {
        "Authorization": f"Bearer {clean_key}",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "VERITAS-ID Hoax Detector",
        "Content-Type": "application/json"
    }

    system_prompt = """Anda adalah analis pakar verifikasi fakta dan ekstraksi klaim berita Indonesia.
Tugas Anda adalah membaca input teks atau naskah berita, lalu mengisinya ke dalam struktur JSON berikut:
{
  "extracted_claim": "1 kalimat deklaratif padat yang menjadi klaim fakta/hoax utama dari teks (maksimal 25 kata).",
  "key_entities": ["Daftar entitas penting seperti nominal uang, angka, tanggal, nama tokoh, atau lokasi yang terdapat dalam teks"],
  "summary": "Ringkasan padat isi teks dalam 2-3 kalimat",
  "sensational_rating": "TINGGI" atau "SEDANG" atau "RENDAH" (berdasarkan tingkat penggunaan nada provokatif/sensasional)
}

PENTING: Kembalikan HANYA format JSON valid tanpa format markdown ```json ... ``` dan tanpa teks tambahan apapun."""

    user_prompt = f"Ekstrak klaim utama dan analisis teks berikut:\n\n{clean_input[:3500]}"

    last_error = ""
    for model_to_use in candidate_models:
        payload = {
            "model": model_to_use,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2
        }

        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=12)
            if response.status_code == 200:
                res_data = response.json()
                choices = res_data.get("choices", [])
                if choices and "message" in choices[0]:
                    content = choices[0]["message"].get("content", "").strip()
                    content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
                    content = re.sub(r'^```\s*', '', content, flags=re.MULTILINE).strip()

                    try:
                        parsed_json = json.loads(content)
                        return {
                            "success": True,
                            "is_llm": True,
                            "provider": "OpenRouter",
                            "extracted_claim": parsed_json.get("extracted_claim", clean_input[:150]),
                            "key_entities": parsed_json.get("key_entities", []),
                            "summary": parsed_json.get("summary", clean_input[:300]),
                            "sensational_rating": parsed_json.get("sensational_rating", "SEDANG"),
                            "model_used": model_to_use,
                            "message": f"✅ Ekstraksi klaim berhasil! (Model: {model_to_use})"
                        }
                    except json.JSONDecodeError:
                        last_error = "Respon tidak berformat JSON valid"
            else:
                last_error = f"HTTP {response.status_code}"
                # Jika terkena rate limit (429) atau model tidak ditemukan (404), 
                # lakukan perulangan untuk mencoba model berikutnya (fallback mechanism).
                # If rate-limited (429) or not found (404), loop to try next model in candidate_models
                continue

        except Exception as e:
            last_error = str(e)
            continue

    res = fallback_heuristic_extraction(clean_input)
    res["message"] = f"⚠️ OpenRouter rate-limited / offline ({last_error}). Menggunakan ekstraksi heuristik lokal."
    return res

# Fungsi uji coba untuk memverifikasi koneksi ke OpenRouter API dan ketersediaan model AI.
def test_openrouter_connection(api_key, model_name=DEFAULT_MODEL):
    """Test function to verify OpenRouter API Key and model availability."""
    clean_key = sanitize_api_key(api_key)
    if not clean_key:
        return {"success": False, "message": "API Key OpenRouter tidak boleh kosong."}

    model = model_name.strip() if model_name and model_name.strip() else DEFAULT_MODEL
    headers = {
        "Authorization": f"Bearer {clean_key}",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "VERITAS-ID Test Connection",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello, respond with OK"}],
        "max_tokens": 10
    }
    try:
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=8)
        if resp.status_code == 200:
            return {"success": True, "message": f"✅ Koneksi OpenRouter API Berhasil! Model '{model}' siap digunakan."}
        else:
            return {"success": False, "message": f"Koneksi OpenRouter Gagal (HTTP {resp.status_code}): {resp.text[:150]}"}
    except Exception as e:
        return {"success": False, "message": f"Gagal terhubung ke OpenRouter API: {str(e)}"}

# Menghasilkan vonis cek fakta dari LLM berdasarkan 5 kategori vonis:
# BENAR, SEBAGIAN BENAR, BELUM ADA BUKTI, SESAT, dan KELIRU.
def get_llm_verdict(raw_text, api_key=None, model_name=DEFAULT_MODEL):
    """
    Get zero-shot LLM fact-checking verdict for raw_text based on 5 standard fact-checking categories:
    1. BENAR: Berdasarkan semua sumber yang ada, pernyataan ini akurat.
    2. SEBAGIAN BENAR: Berdasarkan semua sumber yang secara publik bisa diakses, sebagian pernyataan ini benar.
    3. BELUM ADA BUKTI: Berdasarkan semua bukti yang bisa diperoleh, pernyataan ini tidak bisa disimpulkan akurat atau tidak.
    4. SESAT: Berdasarkan sumber yang ada, pernyataan ini menggunakan fakta dan data yang benar, namun cara penyampaian atau kesimpulannya keliru serta mengarahkan ke tafsir yang salah.
    5. KELIRU: Berdasarkan semua bukti yang ada, pernyataan ini tidak akurat.
    """
    clean_input = raw_text.strip()
    if not clean_input:
        return {
            "success": False,
            "verdict": "BELUM ADA BUKTI",
            "confidence": 0.0,
            "hoax_probability": 50.0,
            "reasoning": "Teks input kosong.",
            "is_llm": False
        }

    clean_key = sanitize_api_key(api_key)

    # Heuristic Fallback Verdict if API key missing
    if not clean_key:
        sensational_words = ["segeralah", "sebarkanlah", "viral", "awas", "bahaya", "ajaib", "gratis", "hadiah", "rahasia", "porcine", "penipuan", "bit.ly", "sinar kosmik", "jam 12"]
        count = sum(1 for w in sensational_words if w in clean_input.lower())
        if count >= 2:
            verdict = "KELIRU"
            conf = 85.0
            hoax_prob = 85.0
            reason = "Terdeteksi pola kata kunci sensasional yang kuat khas pesan hoaks berantai."
        elif count == 1:
            verdict = "SESAT"
            conf = 70.0
            hoax_prob = 70.0
            reason = "Terdeteksi frasa sensasional yang mengarah ke disinformasi/penyesatan publik."
        else:
            verdict = "BELUM ADA BUKTI"
            conf = 50.0
            hoax_prob = 50.0
            reason = "API Key LLM belum dikonfigurasi. Menggunakan vonis heuristik netral."

        return {
            "success": True,
            "verdict": verdict,
            "confidence": conf,
            "hoax_probability": hoax_prob,
            "reasoning": reason,
            "is_llm": False,
            "message": "Menggunakan vonis heuristik lokal (API key tidak diisi)."
        }

    preferred_model = model_name.strip() if model_name and model_name.strip() else DEFAULT_MODEL
    candidate_models = [preferred_model] + [m for m in FREE_MODEL_FALLBACKS if m != preferred_model]

    headers = {
        "Authorization": f"Bearer {clean_key}",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "VERITAS-ID Verdict Engine",
        "Content-Type": "application/json"
    }

    system_prompt = """Anda adalah pakar verifikator fakta independen (fact-checker) profesional di Indonesia.
Tugas Anda adalah mengevaluasi kebenaran klaim/teks yang diberikan dan memberikan VONIS HANYA dalam 1 dari 5 kategori berikut:

1. BENAR -> Berdasarkan semua sumber yang ada, pernyataan ini akurat.
2. SEBAGIAN BENAR -> Berdasarkan semua sumber yang secara publik bisa diakses, sebagian pernyataan ini benar.
3. BELUM ADA BUKTI -> Berdasarkan semua bukti yang bisa diperoleh, pernyataan ini tidak bisa disimpulkan akurat atau tidak.
4. SESAT -> Berdasarkan sumber yang ada, pernyataan ini menggunakan fakta dan data yang benar, namun cara penyampaian atau kesimpulannya keliru serta mengarahkan ke tafsir yang salah.
5. KELIRU -> Berdasarkan semua bukti yang ada, pernyataan ini tidak akurat.

Kembalikan HANYA format JSON valid tanpa format markdown ```json ... ``` dan tanpa teks tambahan apapun:
{
  "verdict": "BENAR" atau "SEBAGIAN BENAR" atau "BELUM ADA BUKTI" atau "SESAT" atau "KELIRU",
  "confidence": angka 0-100 (keyakinan analisis Anda),
  "hoax_probability": angka 0-100 (probabilitas teks ini adalah disinformasi/hoaks/penyesatan. BENAR=5-15, SEBAGIAN BENAR=25-40, BELUM ADA BUKTI=50, SESAT=70-85, KELIRU=85-98),
  "reasoning": "Penjelasan ringkas (2-3 kalimat Bahasa Indonesia) mengenai dasar pengambilan vonis ini."
}"""

    user_prompt = f"Evaluasi dan berikan vonis fakta untuk teks berikut:\n\n{clean_input[:3000]}"

    last_err = ""
    for model_to_use in candidate_models:
        payload = {
            "model": model_to_use,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1
        }

        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=12)
            if response.status_code == 200:
                res_data = response.json()
                choices = res_data.get("choices", [])
                if choices and "message" in choices[0]:
                    content = choices[0]["message"].get("content", "").strip()
                    content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
                    content = re.sub(r'^```\s*', '', content, flags=re.MULTILINE).strip()

                    try:
                        parsed = json.loads(content)
                        v = parsed.get("verdict", "BELUM ADA BUKTI").upper()
                        if v not in ["BENAR", "SEBAGIAN BENAR", "BELUM ADA BUKTI", "SESAT", "KELIRU"]:
                            v = "BELUM ADA BUKTI"

                        return {
                            "success": True,
                            "verdict": v,
                            "confidence": float(parsed.get("confidence", 75.0)),
                            "hoax_probability": float(parsed.get("hoax_probability", 50.0)),
                            "reasoning": parsed.get("reasoning", "Vonis fakta dihasilkan oleh analisis AI."),
                            "is_llm": True,
                            "model_used": model_to_use,
                            "message": f"✅ Vonis LLM berhasil diproses! ({model_to_use})"
                        }
                    except json.JSONDecodeError:
                        last_err = "JSON Decode error"
            else:
                last_err = f"HTTP {response.status_code}"
                continue
        except Exception as e:
            last_err = str(e)
            continue

    # Fallback heuristic if all models fail
    return {
        "success": True,
        "verdict": "BELUM ADA BUKTI",
        "confidence": 50.0,
        "hoax_probability": 50.0,
        "reasoning": f"Seluruh model LLM rate-limited / offline ({last_err}). Menggunakan vonis netral.",
        "is_llm": False
    }

# Aliases for compatibility
test_llm_connection = test_openrouter_connection

