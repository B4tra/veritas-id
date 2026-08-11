# Modul verifikasi silang (cross-check) klaim menggunakan algoritma TF-IDF Cosine Similarity 
# terhadap korpus database lokal, dengan mekanisme fallback ke Google Fact Check API.
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from modules.database import fetch_all_fact_checks

# Kelas utama untuk menangani pencarian kemiripan teks dengan database fakta.
class CrossChecker:
    def __init__(self):
        self.fact_checks = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.load_and_build_index()

    # Memuat data dari database dan membangun matriks vektor TF-IDF untuk perhitungan similarity.
    def load_and_build_index(self):
        """Fetch all fact check records and compute TF-IDF vector matrix."""
        self.fact_checks = fetch_all_fact_checks()
        if not self.fact_checks:
            return

        corpus = []
        for fc in self.fact_checks:
            # Combine title, claim, and content for rich text representation
            text = f"{fc['title']} {fc['claim']} {fc['content']}"
            corpus.append(text.lower())

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    # Melakukan query fallback ke Google Fact Check API jika tidak ada kecocokan yang kuat.
    def query_google_factcheck_api(self, query_text, top_k=2):
        """Fallback query to Google Fact Check Tools REST API."""
        live_matches = []
        api_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "query": query_text[:100],
            "languageCode": "id"
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        try:
            res = requests.get(api_url, params=params, headers=headers, timeout=5)
            if res.status_code == 200:
                data = res.json()
                claims = data.get("claims", [])
                for c in claims[:top_k]:
                    claim_text = c.get("text", "")
                    claim_reviews = c.get("claimReview", [])
                    if claim_reviews:
                        rev = claim_reviews[0]
                        rating = rev.get("textualRating", "Terverifikasi")
                        pub_name = rev.get("publisher", {}).get("name", "Google Fact Check")
                        review_url = rev.get("url", "#")
                        review_title = rev.get("title", claim_text)
                        
                        label = "HOAX" if any(w in rating.lower() for w in ["hoax", "salah", "false", "bohong", "misleading"]) else "FAKTA"
                        
                        live_matches.append({
                            "id": 9999,
                            "title": f"[Live API] {review_title}",
                            "claim": claim_text,
                            "content": claim_text,
                            "label": label,
                            "category": "Live API (Google Fact Check)",
                            "source_name": f"{pub_name} (via Google Fact Check API)",
                            "source_url": review_url,
                            "verdict_details": f"🌐 Rujukan Live Google Fact Check API | Vonis Publisher: {rating}. {review_title}",
                            "similarity_score": 95.0,
                            "is_live_api": True
                        })
        except Exception as e:
            print(f"[Google FactCheck API Error]: {e}")
        return live_matches

    # Mencari kecocokan teks klaim menggunakan TF-IDF Cosine Similarity.
    # Jika kemiripan (similarity) sangat rendah, sistem akan fallback menggunakan Google Fact Check API.
    def find_matches(self, query_text, threshold=0.15, top_k=3):
        """
        Match input query text against fact check corpus using TF-IDF Cosine Similarity.
        If local matches are weak or empty, query Google Fact Check API as fallback.
        """
        if not self.vectorizer or self.tfidf_matrix is None or not self.fact_checks:
            self.load_and_build_index()

        matches = []
        if self.vectorizer and self.tfidf_matrix is not None and self.fact_checks:
            query_vector = self.vectorizer.transform([query_text.lower()])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]

            top_indices = np.argsort(similarities)[::-1]
            for idx in top_indices:
                score = float(similarities[idx])
                if score >= threshold and len(matches) < top_k:
                    record = dict(self.fact_checks[idx])
                    record["similarity_score"] = round(score * 100, 1)
                    record["is_live_api"] = False
                    matches.append(record)

        # Mekanisme fallback ke Google Fact Check API apabila nilai similarity tertinggi terlalu rendah (< 25%)
        # Fallback to Google Fact Check API if top match similarity is low (< 25%)
        if not matches or (matches and matches[0]["similarity_score"] < 25.0):
            api_results = self.query_google_factcheck_api(query_text, top_k=2)
            if api_results:
                matches = api_results + matches

        return matches
