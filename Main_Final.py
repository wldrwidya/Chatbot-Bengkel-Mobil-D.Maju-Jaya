# Main.py (versi logging JSON + simpan semua pertanyaan termasuk skor 0)
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup  # Untuk akses objek Update Telegram dan inline buttons
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, CallbackContext  # Framework bot Telegram
from transformers import AutoTokenizer, AutoModelForQuestionAnswering  # Model QA IndoBERT
from sklearn.feature_extraction.text import TfidfVectorizer  # Untuk representasi kata kunci jadi angka (TF-IDF)
from sklearn.metrics.pairwise import cosine_similarity  # Hitung kemiripan cosine antara pertanyaan dan keyword
import torch  # Operasi tensor untuk model
import json  # Baca/tulis JSON
import os  # Manipulasi path file
import sqlite3  # SQLite untuk penyimpanan dataset & kartu pekerjaan
from datetime import datetime, timedelta  # untuk penjadwalan antrian/tanggal
from typing import List, Tuple
import re  # untuk parsing placeholder

# ========== CONFIG ==========
finetuned_model_path = r"C:\Users\Widya HW\OneDrive - Universitas Tarumanagara\Desktop\Skripsi\Fix Program\indobert-qa-finetuned-final-v2"

# file JSON (fallback jika DB tidak ada)
qa_service_path = r"C:\Users\Widya HW\OneDrive - Universitas Tarumanagara\Desktop\Skripsi\Fix Program\converted_jsons\QA_Final\QA_Service.json"
oli_fix_path = r"C:\Users\Widya HW\OneDrive - Universitas Tarumanagara\Desktop\Skripsi\Fix Program\converted_jsons\QA_Final\oli_fix.json"
gab_umum_path = r"C:\Users\Widya HW\OneDrive - Universitas Tarumanagara\Desktop\Skripsi\Fix Program\converted_jsons\QA_Final\gabungan_umum.json"
gab_bis_path = r"C:\Users\Widya HW\OneDrive - Universitas Tarumanagara\Desktop\Skripsi\Fix Program\converted_jsons\QA_Final\gabungan_bis_truk.json"

# DB path (gunakan ini sebagai sumber data utama)
DB_PATH = r"C:\Users\Widya HW\OneDrive - Universitas Tarumanagara\Desktop\Skripsi\Fix Program\bengkel.db"

TOKEN = "[INPUT TOKEN DARI BOTFATHER]" # Token bot Telegram
ADMIN_CHAT_ID = "[INPUT ID TELEGRAM]"  # ID Telegram admin untuk invoice
TFIDF_THRESHOLD = 0.3  # Ambang batas similarity TF-IDF

# ============================

# === Load model/tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)  # Load tokenizer IndoBERT finetuned
model = AutoModelForQuestionAnswering.from_pretrained(finetuned_model_path) # load model QA
model.eval()  # set model ke mode evaluasi (tidak training)

# === SQLite helpers ===
def init_db_connection(db_path: str) -> sqlite3.Connection:
    """
    NOTE: detect_types disabled intentionally to avoid sqlite auto-conversion issues
    when strings with 'T' (isoformat datetimes) are stored. Treat dates as TEXT.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_tables(conn: sqlite3.Connection):
    """Buat tabel kartu_pekerjaan jika belum ada. Tabel dataset (faq_service, harga_oli, dll) diasumsikan sudah ada."""
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS kartu_pekerjaan (
        id_kartu INTEGER PRIMARY KEY AUTOINCREMENT,
        Nama_pengirim TEXT,
        Merek_mobil TEXT,
        Jenis_mobil TEXT,
        Plat_nomor TEXT,
        Keluhan TEXT,
        tanggal_input TEXT,
        tanggal_datang TEXT,
        nomor_antrian INTEGER,
        status TEXT
    )
    """)
    # Pastikan juga tabel harga_data ada (placeholder -> harga)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS harga_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        placeholder TEXT UNIQUE,
        harga TEXT
    )
    """)
    conn.commit()

def load_keywords_and_paragraphs_from_db(conn: sqlite3.Connection, table_name: str) -> Tuple[List[str], List[dict]]:
    """
    Ambil kolom keyword dan context (disimpan sebagai dict paragraph mirip format JSON lama)
    Jika tabel tidak ada atau kosong, return ([],[])
    """
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT id, category, question, answer, context, keyword FROM {table_name}")
    except Exception:
        return [], []

    rows = cur.fetchall()
    keywords = []
    paragraphs = []
    for r in rows:
        kw = (r["keyword"] or "").strip()
        if kw == "":
            kw = "_nokey"
        keywords.append(kw)
        para = {
            "context": r["context"] or "",
            "qas": [
                {
                    "id": r["id"],
                    "question": r["question"],
                    "answers": [
                        {
                            "text": r["answer"],
                            # answer_start/end tidak diperlukan untuk runtime QA inference, jadi skip jika tidak ada
                        }
                    ]
                }
            ],
            "keyword": kw
        }
        paragraphs.append(para)
    return keywords, paragraphs

# === Harga placeholder helpers (baru) ===
def get_price_by_placeholder(conn: sqlite3.Connection, placeholder_name: str) -> str:
    """
    Ambil harga dari tabel harga_data berdasarkan kolom placeholder.
    Jika tidak ditemukan, kembalikan None.
    """
    cur = conn.cursor()
    cur.execute("SELECT harga FROM harga_data WHERE placeholder = ?", (placeholder_name,))
    r = cur.fetchone()
    if r:
        return r["harga"]
    return None

def replace_placeholders_in_text(conn: sqlite3.Connection, text: str) -> str:
    if not text :
        return text
    
    pattern = r"\{+\s*\{+\s*([A-Za-z0-9 _\-]+?)\s*\}+\s*\}+"

    def repl(match):
        raw = match.group(1)
        # NORMALISASI:
        # hilangkan spasi & underscore berantakan jadi underscore normal
        cleaned = raw.replace(" ", "")
        cleaned = re.sub(r"_+", "_", cleaned)  # rapikan underscore dobel

        # ambil harga dari DB
        price = get_price_by_placeholder(conn, cleaned)
        if price:
            return price
        
        # jika tidak ada di DB, kembalikan placeholder-nya utuh (atau kosong)
        return match.group(0)
    return re.sub(pattern, repl, text)

# === Load datasets (DB preferred; fallback ke JSON) ===
def load_json_safe(path):
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")    # tampilkan warning jika file hilang
        return {"data": []}                        # return kosong biar tidak error
    with open(path, "r", encoding="utf-8") as f:   # buka file JSON
        return json.load(f)                        # parse isinya ke Python dict

# Try to init DB and load keywords; if DB missing or table missing, fallback to JSON
conn = None
try:
    conn = init_db_connection(DB_PATH)
    ensure_tables(conn)
except Exception as e:
    print(f"[WARN] Gagal konek DB: {e}. Akan fallback ke file JSON.")

def load_dataset(table_name: str, json_path: str):
    """Return keywords, paragraphs — diberi prioritas DB kalau tersedia"""
    if conn:
        kws, paras = load_keywords_and_paragraphs_from_db(conn, table_name)
        if kws and paras:
            return kws, paras
    # fallback ke json file
    json_data = load_json_safe(json_path)
    # extract from json_data like sebelumnya
    kws = []
    paras = []
    for item in json_data.get("data", []):
        for para in item.get("paragraphs", []):
            kw = str(para.get("keyword", "")).strip() or "_nokey"
            kws.append(kw)
            paras.append(para)
    return kws, paras

# load datasets (db table names assumed: faq_service, harga_oli, harga_umum_mobil, harga_umum_bis_truk)
kw_qa, para_qa = load_dataset("qa_service", qa_service_path)
kw_oli, para_oli = load_dataset("oli_fix", oli_fix_path)
kw_umum, para_umum = load_dataset("gabungan_umum", gab_umum_path)
kw_bis, para_bis = load_dataset("gabungan_bis_truk", gab_bis_path)

# === Extract keyword & paragraph (fungsi lama dipertahankan jika perlu) ===
def extract_keyword_and_paragraphs_from_json(dataset):
    keywords, paragraphs = [], []
    for item in dataset.get("data", []):
        for para in item.get("paragraphs", []):
            kw = str(para.get("keyword", "")).strip()
            keywords.append(kw if kw else "_nokey")
            paragraphs.append(para)
    return keywords, paragraphs

# === TF-IDF ===
def build_tfidf(keywords):
    # jika keywords kosong (misal DB kosong), beri fallback minimal agar tidak crash
    if not keywords:
        keywords = ["__nokey__"]
    vectorizer = TfidfVectorizer().fit(keywords)          # latih TF-IDF dari semua keyword
    matrix = vectorizer.transform(keywords)               # hasilkan matrix TF-IDF
    return vectorizer, matrix

# Buat TF-IDF untuk bisa digunakan ke 4 mode
vec_qa, mat_qa = build_tfidf(kw_qa)
vec_oli, mat_oli = build_tfidf(kw_oli)
vec_umum, mat_umum = build_tfidf(kw_umum)
vec_bis, mat_bis = build_tfidf(kw_bis)

# === Retrieval function ===
def find_best_keyword_match(question, vectorizer, tfidf_matrix, paragraphs, keywords):
    q_vec = vectorizer.transform([question])              #  ubah pertanyaan jadi vektor TF-IDF
    sims = cosine_similarity(q_vec, tfidf_matrix)         #  hitung kesamaan antar keyword
    best_idx = int(sims.argmax())                         #  ambil index keyword dengan skor tertinggi
    best_score = float(sims[0, best_idx])                 #  simpan skor kesamaannya
    best_kw = keywords[best_idx]                          #  ambil keyword terbaik
    best_para = paragraphs[best_idx]                      #  ambil paragraf yang sesuai
    return best_para, best_kw, best_score, best_idx       #  kembalikan hasil pencarian

# === IndoBERT QA ===
def answer_question_with_model(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)  #  encoding
    with torch.no_grad():                                             #  matikan gradient (evaluasi mode)
        outputs = model(**inputs)                                     #  forward pass model
    start_idx = torch.argmax(outputs.start_logits, dim=-1)[0].item()  #  cari posisi awal jawaban
    end_idx = torch.argmax(outputs.end_logits, dim=-1)[0].item() + 1  #  cari posisi akhir jawaban
    tokens = inputs["input_ids"][0][start_idx:end_idx]                #  ambil token hasil prediksi
    return tokenizer.decode(tokens, skip_special_tokens=True).strip() #  decode ke teks jawaban

# === Logging JSON per baris ===
def save_chat_log(mode, question, answer, keyword, score, best_idx=None, context_text=None):
    """Simpan log dalam format JSON per baris agar bisa dievaluasi otomatis"""
    log_files = {
        "layanan": "chat_log_service.txt",
        "harga_oli": "chat_log_oli.txt",
        "harga_umum_mobil": "chat_log_umum_mobil.txt",
        "harga_umum_bis": "chat_log_bis_truk.txt",
    }

    file_name = log_files.get(mode)   # tentukan file log sesuai mode
    if not file_name:
        return

    log_path = os.path.join(os.path.dirname(__file__), file_name) # buat path lengkap file log

    log_data = {                   # struktur data yang disimpan
        "mode": mode,
        "question": question,
        "prediction": answer,
        "keyword": keyword,
        "tfidf_score": round(score, 3),
        "tfidf_threshold": TFIDF_THRESHOLD,
        "best_index": best_idx,
        "context_used": context_text,
    }
    # simpan log sebagai satu baris JSON per interaksi
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

# === User state ===
user_data = {}   # menyimpan mode aktif tiap user dan state tambahan (Menunggu Plat, Menunggu Konfirmasi, dsb.)

# === Helpers for kartu pekerjaan (antrian) ===
def count_bookings_on_date(conn: sqlite3.Connection, date_obj: datetime.date) -> int:
    """
    Menghitung jumlah booking pada tanggal tertentu. date_obj adalah datetime.date
    Simpan di DB sebagai ISO date string (YYYY-MM-DD).
    """
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as c FROM kartu_pekerjaan WHERE tanggal_datang = ?", (date_obj.isoformat(),))
    r = cur.fetchone()
    # r mungkin None jika tidak ada
    return r["c"] if r and "c" in r.keys() else (r[0] if r else 0)

def schedule_next_available_date_and_position(conn: sqlite3.Connection) -> Tuple[str, int]:
    """
    Logika:
    - Mulai dari hari esok (H+1)
    - Jika hari itu sudah memiliki <5 booking, pakai hari itu dengan nomor_antrian = count+1
    - Jika >=5, maju 1 hari, ulangi
    Return: (tanggal_datang_iso_str, nomor_antrian)
    """
    day = datetime.now().date() + timedelta(days=1)
    while True:
        c = count_bookings_on_date(conn, day)
        if c < 5:
            return day.isoformat(), c + 1
        day = day + timedelta(days=1)

def insert_kartu_pekerjaan(conn: sqlite3.Connection, nama, merek, jenis, plat, keluhan, tanggal_input, tanggal_datang, nomor_antrian, status="Menunggu"):
    """
    tanggal_input: ISO string (YYYY-MM-DD atau datetime iso)
    tanggal_datang: ISO string (YYYY-MM-DD)
    """
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO kartu_pekerjaan (Nama_pengirim, Merek_mobil, Jenis_mobil, Plat_nomor, Keluhan, tanggal_input, tanggal_datang, nomor_antrian, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (nama, merek, jenis, plat, keluhan, tanggal_input, tanggal_datang, nomor_antrian, status))
    conn.commit()
    return cur.lastrowid

def fetch_kartu_by_id(conn: sqlite3.Connection, id_kartu: int):
    cur = conn.cursor()
    cur.execute("SELECT * FROM kartu_pekerjaan WHERE id_kartu = ?", (id_kartu,))
    return cur.fetchone()

# === Build Inline Keyboard for /start ===
def build_start_keyboard():
    keyboard = [
        [InlineKeyboardButton("1. Tanya Layanan Bengkel", callback_data="mode_1")],
        [InlineKeyboardButton("2. Pencatatan Keluhan - Buat Kartu Pekerjaan", callback_data="mode_2")],
        [InlineKeyboardButton("3. Tanya Harga Oli", callback_data="mode_3")],
        [InlineKeyboardButton("4. Tanya Harga Jasa Mobil", callback_data="mode_4")],
        [InlineKeyboardButton("5. Tanya Harga Jasa Bis/Truk", callback_data="mode_5")],
    ]
    return InlineKeyboardMarkup(keyboard)

# === /start ===
async def start(update: Update, context: CallbackContext):
    # tampilkan pesan pembuka dan inline buttons
    await update.message.reply_text(
        "Halo! Selamat datang di Bengkel D. Maju Jaya 🚗\n\n"
        "Pilih layanan yang ingin kamu gunakan:",
        reply_markup=build_start_keyboard()
    )

# CallbackQuery handler (untuk inline buttons)
async def callback_query_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()
    data = query.data

    chat_id = query.message.chat.id
    # map callback_data ke mode
    if data == "mode_1":
        user_data[chat_id] = {"mode": "layanan"}
        await query.message.reply_text(
            "Berikut merupakan layanan yang tersedia di Bengkel D. Maju Jaya:\n"
            "- Ganti Ban\n"
            "- Fogging AC\n"
            "- Overhaul\n"
            "- Flushing AC\n"
            "- Service Rem\n"
            "- Spooring\n"
            "- Balancing\n"
            "- Ganti Oli\n"
            "- Carbon Cleaning\n"
            "- Tune Up Mesin\n\n"
            "Silakan ketik pertanyaanmu tentang layanan di atas:"
        )
        return

    if data == "mode_2":
        user_data[chat_id] = {"mode": "keluhan"}
        # Kirim template langsung supaya user tinggal isi
        template = (
            "Silakan isi template berikut (salin & isi di kotak chat lalu kirim):\n\n"
            "Nama: \n"
            "Merek Mobil: \n"
            "Jenis Mobil: \n"
            "Keluhan: \n\n"
            "Contoh, Merek Mobil: Toyota, Jenis Mobil: Avanza"
        )
        await query.message.reply_text(template)
        return

    if data == "mode_3":
        user_data[chat_id] = {"mode": "harga_oli"}
        await query.message.reply_text(
            "🛢️ Layanan Tanya Harga Oli — kategori tersedia:\n"
            "- Castrol\n"
            "- Repsol\n"
            "- Shell\n"
            "- Pertamina\n"
            "- Top 1\n"
            "- Mobil Oil\n"
            "- Mix Oil\n\n"
            "Silakan ketik pertanyaanmu tentang harga oli di atas:"
        )
        return

    if data == "mode_4":
        user_data[chat_id] = {"mode": "harga_umum_mobil"}
        await query.message.reply_text(
            "💬 Mode 4 — Tanya harga barang/jasa umum untuk kategori *Mobil*.\nSilakan ketik pertanyaan Anda."
        )
        return

    if data == "mode_5":
        user_data[chat_id] = {"mode": "harga_umum_bis"}
        await query.message.reply_text(
            "🚌 Mode 5 — Tanya harga barang/jasa umum untuk kategori *Bis/Truk*.\nSilakan ketik pertanyaan Anda."
        )
        return

# === Message handler ===
async def handle_message(update: Update, context: CallbackContext):
    chat_id = update.message.chat.id     # ambil ID user
    text = update.message.text.strip()   #  ambil teks pesan user

    # === support: if user types /start again, show keyboard (handled by command) ===
    # === MENU PILIHAN via numeric typing (tetap didukung seperti sebelumnya) ===
    if text == "1":
        user_data[chat_id] = {"mode": "layanan"}            # set mode user ke "layanan"
        await update.message.reply_text(                     # menampilkan daftar layanan
            "Berikut merupakan layanan yang tersedia di Bengkel D. Maju Jaya:\n"
            "- Service Ban\n"
            "- Fogging AC\n"
            "- Overhaul\n"
            "- Flushing AC\n"
            "- Service Rem\n"
            "- Spooring\n"
            "- Balancing\n"
            "- Ganti Oli\n"
            "- Carbon Cleaning\n"
            "- Tune Up Mesin\n\n"
            "Silakan ketik pertanyaanmu tentang layanan di atas:"
        )
        return

    if text == "2":
        user_data[chat_id] = {"mode": "keluhan"}
        # Kirim template langsung supaya user tinggal isi
        template = (
            "Silakan isi template berikut (salin & isi di kotak chat lalu kirim):\n\n"
            "Nama: \n"
            "Merek Mobil: \n"
            "Jenis Mobil: \n"
            "Keluhan: \n\n"
            "Contoh, Merek Mobil: Toyota, Jenis Mobil: Avanza"
        )
        await update.message.reply_text(template)
        return

    if text == "3":
        user_data[chat_id] = {"mode": "harga_oli"}
        await update.message.reply_text(
            "🛢️ Layanan Tanya Harga Oli — kategori tersedia:\n"
            "- Castrol\n"
            "- Repsol\n"
            "- Shell\n"
            "- Pertamina\n"
            "- Top 1\n"
            "- Mobil Oil\n"
            "- Mix Oil\n\n"
            "Silakan ketik pertanyaanmu tentang harga oli di atas:"
        )
        return

    if text == "4":
        user_data[chat_id] = {"mode": "harga_umum_mobil"}
        await update.message.reply_text(
            "💬 Mode 4 — Tanya harga barang/jasa umum untuk kategori *Mobil*.\nSilakan ketik pertanyaan Anda."
        )
        return

    if text == "5":
        user_data[chat_id] = {"mode": "harga_umum_bis"}
        await update.message.reply_text(
            "🚌 Mode 5 — Tanya harga barang/jasa umum untuk kategori *Bis/Truk*.\nSilakan ketik pertanyaan Anda."
        )
        return

    # === MODE 1: Layanan ===
    if chat_id in user_data and user_data[chat_id].get("mode") == "layanan":
        para, kw, score, idx = find_best_keyword_match(text, vec_qa, mat_qa, para_qa, kw_qa)
        print(f"[MODE1] Q: {text}\nBest KW: {kw} | Score={score:.3f}")

        if score < TFIDF_THRESHOLD: # jika skor di bawah ambang
            ans = "— Tidak dijawab (skor di bawah threshold) —"
            await update.message.reply_text("Maaf, pertanyaan tidak dapat dipahami.")
            save_chat_log("layanan", text, ans, kw, score, idx, None) # simpan log
            return

        ctx = f"Keyword terkait: {kw}. {para.get('context','')}"       #  ambil context teks
        ans = answer_question_with_model(text, ctx)                    #  jawab pakai IndoBERT

        ans = replace_placeholders_in_text(conn, ans)

        # -----------------------------
        # NOTE: untuk mode layanan (mode 1) TIDAK melakukan placeholder replacement,
        # karena data layanan umumnya tidak mengandung placeholder harga.
        # Jika ada placeholder di context/answer untuk mode 1, dan ingin diganti,
        # kita bisa tambahkan replace_placeholders_in_text(conn, ans) — beri tahu saya.
        # -----------------------------

        await update.message.reply_text(f"💬 {ans or 'Maaf, belum ada jawaban yang sesuai.'}")
        save_chat_log("layanan", text, ans, kw, score, idx, ctx)
        return

    # === MODE 3: Harga Oli ===
    if chat_id in user_data and user_data[chat_id].get("mode") == "harga_oli":
        para, kw, score, idx = find_best_keyword_match(text, vec_oli, mat_oli, para_oli, kw_oli)
        print(f"[MODE3] Q: {text}\nBest KW: {kw} | Score={score:.3f}")

        if score < TFIDF_THRESHOLD:
            ans = "— Tidak dijawab (skor di bawah threshold) —"
            await update.message.reply_text("Maaf, pertanyaan tidak dapat dipahami.")
            save_chat_log("harga_oli", text, ans, kw, score, idx, None)
            return

        ctx = f"Keyword terkait: {kw}. {para.get('context','')}"
        ans = answer_question_with_model(text, ctx)

        # ====== PATCH: replace placeholder(s) in ans using harga_data table ======
        # jika ans mengandung {{placeholder}} -> ganti tiap placeholder dengan nilai dari DB
        ans_with_price = replace_placeholders_in_text(conn, ans) if conn else ans
        # =======================================================================

        await update.message.reply_text(f"💬 {ans_with_price or 'Maaf, belum ada jawaban yang sesuai.'}")
        save_chat_log("harga_oli", text, ans_with_price, kw, score, idx, ctx)
        return

    # === MODE 4: Umum Mobil ===
    if chat_id in user_data and user_data[chat_id].get("mode") == "harga_umum_mobil":
        para, kw, score, idx = find_best_keyword_match(text, vec_umum, mat_umum, para_umum, kw_umum)
        print(f"[MODE4] Q: {text}\nBest KW: {kw} | Score={score:.3f}")

        if score < TFIDF_THRESHOLD:
            ans = "— Tidak dijawab (skor di bawah ambang)" 
            await update.message.reply_text("Maaf, pertanyaan tidak dapat dipahami.")
            save_chat_log("harga_umum_mobil", text, ans, kw, score, idx, None)
            return

        ctx = f"Keyword terkait: {kw}. {para.get('context','')}"
        ans = answer_question_with_model(text, ctx)

        # ====== PATCH: replace placeholder(s) in ans using harga_data table ======
        ans_with_price = replace_placeholders_in_text(conn, ans) if conn else ans
        # =======================================================================

        await update.message.reply_text(f"💬 {ans_with_price or 'Maaf, belum ada jawaban yang sesuai.'}")
        save_chat_log("harga_umum_mobil", text, ans_with_price, kw, score, idx, ctx)
        return

    # === MODE 5: Umum Bis/Truk ===
    if chat_id in user_data and user_data[chat_id].get("mode") == "harga_umum_bis":
        para, kw, score, idx = find_best_keyword_match(text, vec_bis, mat_bis, para_bis, kw_bis)
        print(f"[MODE5] Q: {text}\nBest KW: {kw} | Score={score:.3f}")

        if score < TFIDF_THRESHOLD:
            ans = "— Tidak dijawab (skor di bawah ambang) —"
            await update.message.reply_text("Maaf, pertanyaan tidak dapat dipahami.")
            save_chat_log("harga_umum_bis", text, ans, kw, score, idx, None)
            return

        ctx = f"Keyword terkait: {kw}. {para.get('context','')}"
        ans = answer_question_with_model(text, ctx)

        # ====== PATCH: replace placeholder(s) in ans using harga_data table ======
        ans_with_price = replace_placeholders_in_text(conn, ans) if conn else ans
        # =======================================================================

        await update.message.reply_text(f"💬 {ans_with_price or 'Maaf, belum ada jawaban yang sesuai.'}")
        save_chat_log("harga_umum_bis", text, ans_with_price, kw, score, idx, ctx)
        return

    # === MODE 2: Keluhan (Kartu Pekerjaan) ===
    # Flow:
    # 1) Bot sebelumnya telah mengirim template (Nama:, Merek Mobil:, Jenis Mobil:, Keluhan:)
    # 2) User mengirim filled template -> bot langsung meminta nomor plat (Menunggu Plat = True)
    # 3) User kirim plat -> bot menyimpan Plat, lalu meminta konfirmasi "Ya, Buatkan." (Menunggu Konfirmasi = True)
    # 4) Jika user konfirmasi Ya, Buatkan. -> bot schedule booking, insert ke DB, kirim ringkasan ke admin, kirim konfirmasi ke user (mengambil data dari DB)
    if chat_id in user_data and user_data[chat_id].get("mode") == "keluhan":
        # normalize text for checks but keep original for storage
        text_lower = text.lower()

        # === Jika sedang menunggu konfirmasi "Ya, Buatkan." ===
        if user_data[chat_id].get("MenungguKonfirmasi"):
            if text_lower.replace(",", "").replace(".", "").strip() in ["ya buatkan", "ya, buatkan", "ya buatkan."]:
                
                if not conn:
                    await update.message.reply_text("⚠️ Maaf, terjadi kesalahan server (database tidak tersedia).")
                    return

                data_state = user_data[chat_id]
                nama = data_state.get("Nama")
                merek = data_state.get("Merek Mobil")
                jenis = data_state.get("Jenis Mobil")
                plat = data_state.get("Plat No")
                keluhan = data_state.get("Keluhan")
                tanggal_input = datetime.now().isoformat()  # simpan datetime ISO agar lengkap

                # Tentukan jadwal & nomor antrean (tanggal_datang adalah ISO date 'YYYY-MM-DD')
                tanggal_datang_iso, nomor_antrian = schedule_next_available_date_and_position(conn)

                # Simpan ke DB (tanggal_input dan tanggal_datang sebagai string ISO)
                id_kartu = insert_kartu_pekerjaan(
                    conn, nama, merek, jenis, plat, keluhan,
                    tanggal_input, tanggal_datang_iso, nomor_antrian, status="Menunggu"
                )

                # Ambil ulang dari DB untuk memastikan valid
                rec = fetch_kartu_by_id(conn, id_kartu)

                # === Kirim ke Admin ===
                invoice = (
                    f"📄 Kartu Pekerjaan (ID: {rec['id_kartu']})\n"
                    f"Nama Pengirim : {rec['Nama_pengirim']}\n"
                    f"Merek Mobil   : {rec['Merek_mobil']}\n"
                    f"Jenis Mobil   : {rec['Jenis_mobil']}\n"
                    f"Plat Nomor    : {rec['Plat_nomor']}\n"
                    f"Keluhan       : {rec['Keluhan']}\n"
                    f"Tanggal Input : {rec['tanggal_input']}\n"
                    f"Tanggal Datang: {rec['tanggal_datang']}\n"
                    f"Nomor Antrian : {rec['nomor_antrian']}\n"
                    f"Status        : {rec['status']}"
                )
                await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=invoice)

                # === Kirim Konfirmasi ke User ===
                tanggal = rec["tanggal_datang"]
                antrian = rec["nomor_antrian"]

                # gunakan parse_mode agar tanda *tebal* bekerja; pastikan string di bawah tertutup rapi
                await update.message.reply_text(
                    f"✅ Keluhan berhasil dicatat!\n\n"
                    f"📅 *Tanggal datang:* {tanggal}\n"
                    f"🔢 *Nomor antrean:* {antrian}\n\n"
                    "Silakan membawa kendaraan Anda sesuai jadwal tersebut ya 😊",
                    parse_mode="Markdown"
                )

                # Selesai → reset state
                del user_data[chat_id]
                return
            
            else:
                await update.message.reply_text(
                    "Ketik *Ya, Buatkan.* untuk menyimpan keluhan atau ketik *Batal* untuk membatalkan.",
                    parse_mode="Markdown"
                )
                return

        # jika sedang menunggu plat
        if user_data[chat_id].get("Menunggu Plat"):
            # treat this message as plat
            user_data[chat_id]["Plat No"] = text.strip()
            # setelah plat, minta konfirmasi
            user_data[chat_id]["MenungguKonfirmasi"] = True
            # jangan hapus Menunggu Plat sebab flow sudah selesai tapi kita bisa del
            if "Menunggu Plat" in user_data[chat_id]:
                del user_data[chat_id]["Menunggu Plat"]
            await update.message.reply_text("Apakah Anda ingin saya buatkan kartu pekerjaan sekarang? Jika ya ketik: Ya, Buatkan.")
            return

        # jika belum mengisi template (pertama kali), bot akan mencoba parse template
        if "Nama" not in user_data[chat_id]:
            # parsing yang diharapkan: 4 baris dengan format "Nama: nilai"
            lines = text.split("\n")
            # menerima varian: jika user copy paste template but leaves blank, still require filled entries
            # cari pola "Nama:" "Merek" "Jenis" "Keluhan" di tiap baris
            parsed = {}
            for line in lines:
                if ":" in line:
                    k, v = line.split(":", 1)
                    parsed[k.strip().lower()] = v.strip()
            # cek apakah keempat field ada
            if all(k in parsed for k in ["nama", "merek mobil", "jenis mobil", "keluhan"]):
                # simpan state
                user_data[chat_id]["Nama"] = parsed["nama"]
                user_data[chat_id]["Merek Mobil"] = parsed["merek mobil"]
                user_data[chat_id]["Jenis Mobil"] = parsed["jenis mobil"]
                user_data[chat_id]["Keluhan"] = parsed["keluhan"]
                # setelah mengisi template, langsung minta plat (sesuai flow yang diminta)
                user_data[chat_id]["Menunggu Plat"] = True
                await update.message.reply_text("Terima kasih. Silakan kirim nomor plat kendaraan:")
            else:
                await update.message.reply_text("⚠️ Format tidak dikenali. Pastikan mengisi template seperti:\n\nNama: <nama>\nMerek Mobil: <merek>\nJenis Mobil: <jenis>\nKeluhan: <isi keluhan>\n\n")
            return

    # jika tidak cocok dengan mode apapun
    await update.message.reply_text("Silakan pilih 1–5, klik tombol pada /start, atau ketik /start untuk mulai kembali.")

# === Run bot ===
def main():
    app = Application.builder().token(TOKEN).build() #  inisialisasi bot
    app.add_handler(CommandHandler("start", start)) #  tangani perintah /start
    app.add_handler(CallbackQueryHandler(callback_query_handler))  # tangani inline button callback
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)) #  tangani semua teks biasa
    print("🚀 Bot Chatbot D.Maju Jaya siap dijalankan di Telegram!")
    app.run_polling()      #  jalankan polling Telegram (loop utama)

if __name__ == "__main__":
    # pastikan koneksi db ready jika tersedia
    if conn:
        ensure_tables(conn)
    main()
