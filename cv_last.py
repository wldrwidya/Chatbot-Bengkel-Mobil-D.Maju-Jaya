import os
import re
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import unicodedata

# ======================
# REGEX
# ======================
PRICE_RE = re.compile(r"(Rp\.?\s*[0-9][0-9\.\,]*)", flags=re.IGNORECASE)
SIZE_RE = re.compile(r"(\d+)\s*(lt|liter|ltr|l)\b", flags=re.IGNORECASE)

# ======================
# HELPERS
# ======================

def slugify(text):
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower())
    return text.strip("_")

def extract_size_near(text, price_span):
    start, end = price_span
    probe = text[max(0, start-20): end+20]
    m = SIZE_RE.search(probe)
    if not m:
        return None
    return m.group(1).lower() + "lt"

# ============================
# PRODUK SLUG DARI QUESTION
# ============================
QUESTION_BLACKLIST = {
    "berapa", "harga", "oli", "infokan", "ongkos", "biaya", "tunjukkan",
    "kasih", "berapaan", "berapa?", "berapa,", "oli", "castrol",
}

def extract_product_slug_from_question(question, category=None):
    q = question.lower()

    # hapus kata-kata sampah
    for blk in QUESTION_BLACKLIST:
        q = q.replace(blk, " ")

    # bersihkan simbol
    q = re.sub(r"[^a-zA-Z0-9]+", " ", q).strip()

    # contoh hasil: "magnatec 10w40"
    tokens = q.split()

    # hapus category dulu jika muncul di depan (Castrol, Shell, dll.)
    if category:
        tokens = [t for t in tokens if t != category.lower()]

    # ambil maksimal 4 token
    product_slug = "_".join(tokens[:4])

    return slugify(product_slug)


def build_placeholder(category, product_slug, size):
    return f"harga_{slugify(category)}_{product_slug}_{size}"


# ======================
# UNIVERSAL PROCESSOR
# ======================
def process_all_excels(input_dir, output_dir):
    harga_dict = defaultdict(lambda: {"harga": None, "row_ids": set(), "source": None})

    files = [f for f in os.listdir(input_dir) if f.endswith(".xlsx")]

    for fname in files:
        print(f"\n=== MEMPROSES {fname} ===")
        path = os.path.join(input_dir, fname)

        df = pd.read_excel(path, dtype=str).fillna("")
        cols = {c.lower(): c for c in df.columns}

        req = ["id","category","question","answer","context","keyword"]
        if any(r not in cols for r in req):
            print("❌ Kolom wajib tidak lengkap → SKIP:", df.columns)
            continue

        grouped = df.groupby(cols["category"])
        squad_data = []

        for category, group in grouped:
            entry = {"title": str(category), "paragraphs": []}

            for _, row in group.iterrows():
                qid = str(row[cols["id"]]).strip()
                question = str(row[cols["question"]]).strip()
                answer_raw = str(row[cols["answer"]]).strip()
                context = str(row[cols["context"]]).strip()
                keyword = str(row[cols["keyword"]]).strip()

                # ===== Tambahkan keyword di awal context =====
                if keyword and not context.lower().startswith("[keyword:"):
                    kw = ", ".join([k.strip() for k in re.split(r"[;,]", keyword) if k.strip()])
                    context = f"[Keyword: {kw}] {context}"

                # ============================
                # DETEKSI PRODUK
                # ============================
                product_slug = extract_product_slug_from_question(question, category)

                # ============================
                # DETEKSI HARGA
                # ============================
                matches = list(PRICE_RE.finditer(answer_raw))

                ph_answer = answer_raw  # akan diganti placeholder

                # CASE A: tidak ada harga → QA NORMAL
                if not matches:
                    # tetap dicatat sebagai QA normal
                    pass

                # CASE B: 1 harga
                elif len(matches) == 1:
                    m = matches[0]
                    price = m.group(0)

                    size = extract_size_near(answer_raw, m.span()) or "1lt"
                    placeholder = build_placeholder(category, product_slug, size)

                    # replace answer
                    ph_answer = ph_answer.replace(price, f"{{{{{placeholder}}}}}")

                    # replace context juga kalau ada
                    if price in context:
                        context = context.replace(price, f"{{{{{placeholder}}}}}")

                    harga_dict[placeholder]["harga"] = price
                    harga_dict[placeholder]["row_ids"].add(qid)
                    harga_dict[placeholder]["source"] = fname

                # CASE C: 2 harga → 1lt & 4lt
                elif len(matches) == 2:
                    m1, m2 = matches
                    price1, price2 = m1.group(0), m2.group(0)

                    placeholder1 = build_placeholder(category, product_slug, "1lt")
                    placeholder2 = build_placeholder(category, product_slug, "4lt")

                    # replace sequential
                    ph_answer = ph_answer.replace(price1, f"{{{{{placeholder1}}}}}", 1)
                    ph_answer = ph_answer.replace(price2, f"{{{{{placeholder2}}}}}", 1)

                    if price1 in context:
                        context = context.replace(price1, f"{{{{{placeholder1}}}}}")
                    if price2 in context:
                        context = context.replace(price2, f"{{{{{placeholder2}}}}}")

                    harga_dict[placeholder1]["harga"] = price1
                    harga_dict[placeholder1]["row_ids"].add(qid)
                    harga_dict[placeholder1]["source"] = fname

                    harga_dict[placeholder2]["harga"] = price2
                    harga_dict[placeholder2]["row_ids"].add(qid)
                    harga_dict[placeholder2]["source"] = fname

                # ============================
                # HITUNG SPAN SETELAH PATCH
                # ============================
                # Kini answer = ph_answer
                # Kita cari posisi ph_answer di context
                if ph_answer in context:
                    a_start = context.index(ph_answer)
                    a_end = a_start + len(ph_answer)
                else:
                    # fallback: cari potongan 15 char pertama
                    frag = ph_answer[:15]
                    if frag in context:
                        a_start = context.index(frag)
                        a_end = a_start + len(frag)
                    else:
                        print("⚠ ANSWER tidak ditemukan di context → SKIP id", qid)
                        continue

                # ============================
                # ADD PARAGRAPH
                # ============================
                entry["paragraphs"].append({
                    "context": context,
                    "qas": [
                        {
                            "id": qid,
                            "question": question,
                            "answers": [
                                {
                                    "text": ph_answer,
                                    "answer_start": a_start,
                                    "answer_end": a_end
                                }
                            ]
                        }
                    ]
                })

            if entry["paragraphs"]:
                squad_data.append(entry)

        # SIMPAN JSON
        out_path = os.path.join(output_dir, Path(fname).stem + ".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"data": squad_data}, f, indent=2, ensure_ascii=False)

        print("✔ Selesai membuat:", out_path)

    # ======================
    # SIMPAN harga_data.json
    # ======================
    harga_list = []
    for ph, d in harga_dict.items():
        harga_list.append({
            "placeholder": ph,
            "harga": d["harga"],
            "source_file": d["source"],
            "row_id": ", ".join(sorted(d["row_ids"]))
        })

    hpath = os.path.join(output_dir, "harga_data.json")
    with open(hpath, "w", encoding="utf-8") as f:
        json.dump(harga_list, f, indent=2, ensure_ascii=False)

    print("\n=== harga_data.json DONE ===")
    print(hpath)


# ======================
# RUN
# ======================
INPUT = r"C:\Users\Widya HW\OneDrive - Universitas Tarumanagara\Desktop\Skripsi\Fix Program\Fix_Excel"
OUTPUT = INPUT
process_all_excels(INPUT, OUTPUT)
