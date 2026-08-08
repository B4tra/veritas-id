import re
import requests
from bs4 import BeautifulSoup

def is_valid_url(text):
    """Check if the given string is a HTTP/HTTPS URL."""
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' # domain...
        r'localhost|' # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(url_pattern.match(text.strip()))

def extract_content_from_url(url):
    """
    Extract title and main article text from news URL.
    Returns dict with title, body, status, error.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url.strip(), headers=headers, timeout=7)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            script.extract()
            
        # Get title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find('h1'):
            title = soup.find('h1').get_text().strip()
            
        # Get main article text paragraphs
        paragraphs = [p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip()) > 30]
        article_text = " ".join(paragraphs)
        
        if not article_text:
            # Fallback to general text
            article_text = soup.get_text(separator=' ', strip=True)
            
        # Truncate if extremely long
        if len(article_text) > 3000:
            article_text = article_text[:3000] + "..."
            
        return {
            "success": True,
            "title": title or "Artikel Berita",
            "text": f"{title}. {article_text}".strip(),
            "body": article_text,
            "url": url,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "title": None,
            "text": None,
            "body": None,
            "url": url,
            "error": f"Gagal mengekstrak konten dari tautan: {str(e)}"
        }
