# Modul web scraping untuk mengambil artikel berita dari 3 sumber:
# - TurnBackHoax.id untuk artikel berlabel HOAX
# - CNN Indonesia untuk artikel berlabel FAKTA
# - ANTARA News untuk artikel berlabel FAKTA
import requests
from bs4 import BeautifulSoup
import sqlite3
import re
import os
import time
import urllib3

# Nonaktifkan peringatan sertifikat SSL yang tidak aman
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fact_check.db")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7"
}

# Fungsi untuk membersihkan teks dari tag HTML dan spasi berlebih
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Fungsi scraper untuk mengambil artikel HOAX dari TurnBackHoax.id
def scrape_turnbackhoax(limit=5):
    """Scrape real live hoax articles from TurnBackHoax.id"""
    items = []
    url = "https://turnbackhoax.id/"
    today_str = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        res = requests.get(url, headers=HEADERS, verify=False, timeout=8)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            seen_titles = set()
            
            links = soup.find_all("a", href=lambda h: h and "/articles/" in h)
            for a in links:
                if len(items) >= limit:
                    break
                
                title = clean_text(a.get_text())
                href = a["href"]
                if not href.startswith("http"):
                    href = "https://turnbackhoax.id" + href
                    
                if not title or len(title) < 20 or "Lihat Semua" in title or title in seen_titles:
                    continue
                    
                seen_titles.add(title)
                
                parent = a.find_parent(["article", "div", "li"])
                snippet = ""
                pub_date = today_str
                if parent:
                    p = parent.find("p")
                    if p:
                        snippet = clean_text(p.get_text())
                    time_elem = parent.find("time") or parent.find("span", class_=lambda c: c and "date" in str(c).lower())
                    if time_elem:
                        pub_date = clean_text(time_elem.get_text())
                        
                verdict = f"SALAH / HOAX. Artikel terverifikasi dari TurnBackHoax.id. {snippet[:200]}"
                
                items.append({
                    "title": title,
                    "claim": title,
                    "content": snippet if len(snippet) > 15 else title,
                    "label": "HOAX",
                    "category": "Hoax Media Sosial",
                    "source_name": "TurnBackHoax.id (MAFINDO)",
                    "source_url": href,
                    "verdict_details": verdict,
                    "published_at": pub_date
                })
    except Exception as e:
        print(f"[Scraper Error] TurnBackHoax: {e}")
    return items

# Fungsi scraper untuk mengambil artikel FAKTA dari CNN Indonesia
def scrape_cnn_indonesia(limit=5):
    """Scrape real live verified news articles from CNN Indonesia"""
    items = []
    url = "https://www.cnnindonesia.com/nasional"
    today_str = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        res = requests.get(url, headers=HEADERS, verify=False, timeout=8)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            articles = soup.find_all("article")
            seen_titles = set()
            
            for a in articles:
                if len(items) >= limit:
                    break
                    
                title_elem = a.find("h2") or a.find("h3")
                if not title_elem:
                    continue
                    
                title = clean_text(title_elem.get_text())
                link_elem = a.find("a")
                href = link_elem["href"] if link_elem and "href" in link_elem.attrs else url
                
                if not title or len(title) < 20 or title in seen_titles:
                    continue
                    
                seen_titles.add(title)
                
                items.append({
                    "title": f"[CNN] {title}",
                    "claim": title,
                    "content": title,
                    "label": "FAKTA",
                    "category": "Berita Nasional CNN",
                    "source_name": "CNN Indonesia",
                    "source_url": href,
                    "verdict_details": "TERVERIFIKASI FAKTA. Liputan jurnalistik resmi dari CNN Indonesia.",
                    "published_at": today_str
                })
    except Exception as e:
        print(f"[Scraper Error] CNN Indonesia: {e}")
    return items

# Fungsi scraper untuk mengambil artikel FAKTA dari ANTARA News
def scrape_antara_news(limit=5):
    """Scrape real live verified news articles from LKBN ANTARA News"""
    items = []
    url = "https://www.antaranews.com/top-news"
    today_str = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        res = requests.get(url, headers=HEADERS, verify=False, timeout=8)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            articles = soup.find_all("article") or soup.find_all("h3")
            seen_titles = set()
            
            for a in articles:
                if len(items) >= limit:
                    break
                link_elem = a.find("a") if hasattr(a, "find") else a
                if not link_elem:
                    continue
                title = clean_text(link_elem.get_text())
                href = link_elem.get("href", url)
                if not href.startswith("http"):
                    href = "https://www.antaranews.com" + href
                    
                if not title or len(title) < 20 or title in seen_titles:
                    continue
                    
                seen_titles.add(title)
                
                items.append({
                    "title": f"[Antara] {title}",
                    "claim": title,
                    "content": title,
                    "label": "FAKTA",
                    "category": "Berita Resmi ANTARA",
                    "source_name": "ANTARA News",
                    "source_url": href,
                    "verdict_details": "TERVERIFIKASI FAKTA. Kantor Berita Indonesia (LKBN ANTARA).",
                    "published_at": today_str
                })
    except Exception as e:
        print(f"[Scraper Error] ANTARA: {e}")
    return items

# Pipeline utama untuk menjalankan seluruh proses web scraping
def run_scraping_pipeline(limit_per_source=5):
    """
    Execute full scraping pipeline for TurnBackHoax (HOAX), CNN Indonesia (FAKTA), and ANTARA News (FAKTA).
    Saves clean records directly to SQLite fact_check.db.
    """
    print("[VERITAS-ID Pipeline] Running live web scraping...")
    tbh_items = scrape_turnbackhoax(limit=limit_per_source)
    cnn_items = scrape_cnn_indonesia(limit=limit_per_source)
    antara_items = scrape_antara_news(limit=limit_per_source)
    
    all_new_items = tbh_items + cnn_items + antara_items
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    inserted_count = 0
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    for item in all_new_items:
        # Logika deduplikasi: mengecek apakah artikel dengan judul, klaim, atau URL yang sama sudah ada di database
        cursor.execute("SELECT COUNT(*) FROM fact_checks WHERE title = ? OR claim = ? OR (source_url IS NOT NULL AND source_url != '' AND source_url = ?)", (item["title"], item["claim"], item["source_url"]))
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO fact_checks (title, claim, content, label, category, source_name, source_url, verdict_details, published_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item["title"], item["claim"], item["content"], item["label"],
                item["category"], item["source_name"], item["source_url"], item["verdict_details"],
                item.get("published_at", now_str), now_str
            ))
            inserted_count += 1
            
    conn.commit()
    conn.close()
    
    return {
        "status": "success",
        "total_scraped": len(all_new_items),
        "inserted_count": inserted_count,
        "tbh_count": len(tbh_items),
        "cnn_count": len(cnn_items),
        "antara_count": len(antara_items),
        "items_sample": [item["title"] for item in all_new_items[:5]]
    }

if __name__ == "__main__":
    res = run_scraping_pipeline(limit_per_source=3)
    print("Live Scrape Output:", res)
