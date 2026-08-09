import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time

def send_gmail_report(sender_email, app_password, recipient_email, report_data):
    """
    Send an HTML formatted daily report email via Gmail SMTP.
    """
    if not sender_email or not app_password or not recipient_email:
        return {"success": False, "message": "Email pengirim, App Password, dan email penerima wajib diisi."}
        
    try:
        # Create message container
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🛡️ [VERITAS-ID] Laporan Otomatis Deteksi Hoax & Cek Fakta ({time.strftime('%d %B %Y')})"
        msg["From"] = f"VERITAS-ID Bot <{sender_email}>"
        msg["To"] = recipient_email
        
        # Fetch total database metrics from SQLite
        try:
            from modules.database import fetch_all_fact_checks
            all_records = fetch_all_fact_checks()
            total_db = len(all_records)
            hoax_db = len([r for r in all_records if r.get("label") == "HOAX"])
            fakta_db = len([r for r in all_records if r.get("label") == "FAKTA"])
        except Exception:
            total_db = 0
            hoax_db = 0
            fakta_db = 0

        # Incremental daily scraping statistics
        inserted = report_data.get("inserted_count", 0)
        tbh_cnt = report_data.get("tbh_count", 0)
        cnn_cnt = report_data.get("cnn_count", 0)
        antara_cnt = report_data.get("antara_count", 0)
        fakta_new_cnt = cnn_cnt + antara_cnt
        items_sample = report_data.get("items_sample", [])
        
        # Build items table HTML for newly scraped articles
        items_html = ""
        if items_sample:
            for idx, item_title in enumerate(items_sample, 1):
                badge = "🔴 HOAX" if "[PENIPUAN]" in item_title or "[SALAH]" in item_title or "TurnBackHoax" in item_title or "HOAX" in item_title else "🟢 FAKTA"
                items_html += f"""
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 10px; font-weight: bold; width: 40px;">{idx}</td>
                    <td style="padding: 10px; width: 100px;">{badge}</td>
                    <td style="padding: 10px;">{item_title}</td>
                </tr>
                """
        else:
            items_html = '<tr><td colspan="3" style="padding: 15px; text-align: center; color: #718096; background: #f7fafc;">Semua berita terkini sudah up-to-date. Tidak ada artikel duplikat baru yang ditambahkan pada pemutakhiran ini.</td></tr>'

        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f7fafc; margin: 0; padding: 20px; }}
                .container {{ max-width: 650px; background: #ffffff; margin: 0 auto; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.08); border: 1px solid #e2e8f0; }}
                .header {{ background: linear-gradient(135deg, #1e1e38 0%, #0d1117 100%); color: #ffffff; padding: 25px; text-align: center; }}
                .header h1 {{ margin: 0; font-size: 24px; font-weight: 800; }}
                .header p {{ margin-top: 5px; color: #a0aec0; font-size: 14px; }}
                .content {{ padding: 25px; color: #2d3748; }}
                .sec-title {{ color: #2b6cb0; font-size: 16px; font-weight: 700; margin-top: 10px; margin-bottom: 12px; }}
                .stat-grid {{ display: table; width: 100%; margin-bottom: 15px; }}
                .stat-cell-scraping {{ display: table-cell; width: 33%; text-align: center; background: #ebf8ff; padding: 12px; border-radius: 8px; border: 1px solid #bee3f8; }}
                .stat-cell-db {{ display: table-cell; width: 33%; text-align: center; background: #f7fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; margin-bottom: 20px; }}
                th {{ background: #edf2f7; padding: 10px; text-align: left; font-size: 13px; color: #4a5568; }}
                .divider {{ border-top: 2px dashed #e2e8f0; margin: 25px 0; }}
                .footer {{ background: #f7fafc; text-align: center; padding: 15px; font-size: 12px; color: #a0aec0; border-top: 1px solid #edf2f7; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🛡️ VERITAS-ID</h1>
                    <p>Laporan Pemutakhiran Otomatis Web Scraper & Deteksi Hoax Harian</p>
                </div>
                <div class="content">
                    
                    <!-- BAGIAN 1: INFORMASI ARTIKEL HASIL SCRAPING TERBARU -->
                    <div class="sec-title">🔄 1. Hasil Pemutakhiran Web Scraping Terbaru ({time.strftime('%Y-%m-%d %H:%M:%S')})</div>
                    <div class="stat-grid">
                        <div class="stat-cell-scraping">
                            <span style="font-size: 22px; font-weight: bold; color: #2b6cb0;">{inserted}</span><br>
                            <small style="color: #4a5568;">Artikel Baru Ter-scrape</small>
                        </div>
                        <div class="stat-cell-scraping">
                            <span style="font-size: 22px; font-weight: bold; color: #e53e3e;">{tbh_cnt}</span><br>
                            <small style="color: #4a5568;">Hoax Baru (TurnBackHoax)</small>
                        </div>
                        <div class="stat-cell-scraping">
                            <span style="font-size: 22px; font-weight: bold; color: #38a169;">{fakta_new_cnt}</span><br>
                            <small style="color: #4a5568;">Fakta Baru (CNN & Antara)</small>
                        </div>
                    </div>

                    <h4 style="margin-bottom: 8px; color: #2d3748;">📰 Rincian Artikel Hasil Scraping Terbaru Ditambahkan:</h4>
                    <table>
                        <thead>
                            <tr>
                                <th>No</th>
                                <th>Label</th>
                                <th>Judul Artikel / Klaim</th>
                            </tr>
                        </thead>
                        <tbody>
                            {items_html}
                        </tbody>
                    </table>

                    <div class="divider"></div>

                    <!-- BAGIAN 2: INFORMASI TOTAL KESELURUHAN DATABASE -->
                    <div class="sec-title">📊 2. Status Total Akumulasi Database Saat Ini</div>
                    <div class="stat-grid">
                        <div class="stat-cell-db">
                            <span style="font-size: 22px; font-weight: bold; color: #2b6cb0;">{total_db}</span><br>
                            <small style="color: #718096;">Total Data Cek Fakta</small>
                        </div>
                        <div class="stat-cell-db">
                            <span style="font-size: 22px; font-weight: bold; color: #e53e3e;">{hoax_db}</span><br>
                            <small style="color: #718096;">Jumlah Data HOAX</small>
                        </div>
                        <div class="stat-cell-db">
                            <span style="font-size: 22px; font-weight: bold; color: #38a169;">{fakta_db}</span><br>
                            <small style="color: #718096;">Jumlah Data FAKTA</small>
                        </div>
                    </div>

                    <p style="margin-top: 20px; font-size: 13px; color: #718096;">
                        💡 <em>Model NLP (Logistic Regression) & Deep Learning (LSTM) telah dilatih ulang secara otomatis menggunakan total korpus data terbaru ini.</em>
                    </p>
                </div>
                <div class="footer">
                    VERITAS-ID • Sistem Deteksi Hoax NLP Indonesia (GEMASTIK XIX 2026)
                </div>
            </div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, "html"))
        
        # Connect to Gmail SMTP Server via TLS
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=15)
        server.starttls()
        server.login(sender_email.strip(), app_password.strip())
        server.sendmail(sender_email.strip(), recipient_email.strip(), msg.as_string())
        server.quit()
        
        return {"success": True, "message": f"Email laporan berhasil dikirim ke {recipient_email}!"}
        
    except Exception as e:
        return {"success": False, "message": f"Gagal mengirim email: {str(e)}"}

def send_test_email(sender_email, app_password, recipient_email):
    """Send a quick test email to verify Gmail credentials."""
    test_data = {
        "inserted_count": 1,
        "total_scraped": 3,
        "tbh_count": 1,
        "cnn_count": 1,
        "antara_count": 1,
        "items_sample": ["[UJI COBA] Sistem Notifikasi Email VERITAS-ID Berhasil Terhubung!"]
    }
    return send_gmail_report(sender_email, app_password, recipient_email, test_data)
