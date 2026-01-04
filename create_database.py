import sqlite3
import json
import os
import re

# ============================
# 1. PATH FOLDER & FILE
# ============================
db_path = r"C:\Users\Widya HW\OneDrive - Universitas Tarumanagara\Desktop\Skripsi\Fix Program\bengkel copy.db"
json_folder = r"C:\Users\Widya HW\OneDrive - Universitas Tarumanagara\Desktop\Skripsi\Fix Program\Fix_Excel\final"

# File JSON SQuAD asli
files_original = {
    "qa_service": "Qa_Service.json",
    "oli_fix": "Oli_Fix.json",
    "umum": "Umum.json",
    "jasa_umum": "Jasa_umum.json",
    "bis_truk": "Bis_Truk.json",
    "jasa_bis_truk": "Jasa_Bis_Truk.json"
}

# Gabungan (target tabel)
merge_map = {
    "gabungan_umum": ["Umum.json", "Jasa_umum.json"],
    "gabungan_bis_truk": ["Bis_Truk.json", "Jasa_Bis_Truk.json"],
    "qa_service": ["Qa_Service.json"],
    "oli_fix": ["Oli_Fix.json"]
}

harga_file = "harga_data.json"


# ============================
# 2. CONNECT DATABASE
# ============================
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("🔌 Terhubung ke database:", db_path)


# ============================
# 3. CREATE TABLE
# ============================

# Tabel gabungan diberi struktur baru (ID auto)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS gabungan_umum (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT,
        question TEXT,
        answer TEXT,
        context TEXT,
        keyword TEXT,
        source_file TEXT,
        original_id TEXT
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS gabungan_bis_truk (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT,
        question TEXT,
        answer TEXT,
        context TEXT,
        keyword TEXT,
        source_file TEXT,
        original_id TEXT
    )
""")

# Tabel qa_service & oli_fix tetap seperti sebelumnya — tidak disentuh
cursor.execute("""
    CREATE TABLE IF NOT EXISTS qa_service (
        id TEXT PRIMARY KEY,
        category TEXT,
        question TEXT,
        answer TEXT,
        context TEXT,
        keyword TEXT
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS oli_fix (
        id TEXT PRIMARY KEY,
        category TEXT,
        question TEXT,
        answer TEXT,
        context TEXT,
        keyword TEXT
    )
""")

# Tabel harga_data tetap seperti sebelumnya — tidak disentuh
cursor.execute("""
    CREATE TABLE IF NOT EXISTS harga_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        placeholder TEXT,
        harga TEXT,
        source_file TEXT
    )
""")

conn.commit()


# ==========================================
# 4. FUNCTION EXTRACT KEYWORD DARI CONTEXT
# ==========================================
def extract_keywords(context):
    match = re.search(r"\[Keyword:\s*(.*?)\]", context)
    if match:
        return match.group(1).strip()
    return ""


# ====================================================
# 5. IMPORT SQuAD UNTUK TABEL GABUNGAN (ID AUTO)
# ====================================================
def import_squad_gabungan(json_files, table_name):
    for json_file in json_files:
        path = os.path.join(json_folder, json_file)
        print(f"📥 Import (gabungan): {json_file}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data["data"]:
            category = item.get("title", "")

            for paragraph in item.get("paragraphs", []):
                context = paragraph.get("context", "")
                keyword = extract_keywords(context)

                for qa in paragraph.get("qas", []):
                    original_id = qa.get("id")
                    question = qa.get("question", "")
                    answer = qa["answers"][0]["text"]

                    cursor.execute(f"""
                        INSERT INTO {table_name}
                        (category, question, answer, context, keyword, source_file, original_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (category, question, answer, context, keyword, json_file, original_id))

        conn.commit()
        print(f"✅ Sukses → {table_name} ({json_file})")


# ====================================================
# 6. IMPORT SQuAD UNTUK qa_service & oli_fix (TETAP)
# ====================================================
def import_squad_original(json_file, table_name):
    path = os.path.join(json_folder, json_file)
    print(f"📥 Import: {json_file}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data["data"]:
        category = item.get("title", "")

        for paragraph in item.get("paragraphs", []):
            context = paragraph.get("context", "")
            keyword = extract_keywords(context)

            for qa in paragraph.get("qas", []):
                qid = qa.get("id")
                question = qa.get("question", "")
                answer = qa["answers"][0]["text"]

                cursor.execute(f"""
                    INSERT OR REPLACE INTO {table_name}
                    (id, category, question, answer, context, keyword)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (qid, category, question, answer, context, keyword))

    conn.commit()
    print(f"✅ Sukses → {table_name}")


# ============================
# 7. JALANKAN IMPORT SQuAD
# ============================
# gabungan
import_squad_gabungan(merge_map["gabungan_umum"], "gabungan_umum")
import_squad_gabungan(merge_map["gabungan_bis_truk"], "gabungan_bis_truk")

# original (tidak diubah)
import_squad_original("Qa_Service.json", "qa_service")
import_squad_original("Oli_Fix.json", "oli_fix")


# ============================
# 8. IMPORT harga_data.json (TETAP)
# ============================
print("\n📥 Import harga_data.json")

harga_path = os.path.join(json_folder, harga_file)

with open(harga_path, "r", encoding="utf-8") as f:
    harga_data = json.load(f)

for item in harga_data:
    placeholder = item.get("placeholder", "")
    harga = item.get("harga", "")
    source_file = item.get("source_file", "")

    cursor.execute("""
        INSERT INTO harga_data (placeholder, harga, source_file)
        VALUES (?, ?, ?)
    """, (placeholder, harga, source_file))

conn.commit()
print("💰 Selesai import harga_data!")


# ============================
# 9. FINISH
# ============================
conn.close()
print("\n🎉 SEMUA DATA BERHASIL DIMASUKKAN KE bengkel.db!")



# # untuk v1 / pembuatan database dari 0
# import sqlite3
# import json
# import os
# import re

# # ============================
# # 1. PATH FOLDER & FILE
# # ============================
# db_path = r"C:\Users\Widya HW\OneDrive - Universitas Tarumanagara\Desktop\Skripsi\Fix Program\bengkel.db"
# json_folder = r"C:\Users\Widya HW\OneDrive - Universitas Tarumanagara\Desktop\Skripsi\Fix Program\Fix_Excel\final"

# # File JSON SQuAD (asli)
# files_original = {
#     "qa_service": "Qa_Service.json",
#     "oli_fix": "Oli_Fix.json",
#     "umum": "Umum.json",
#     "jasa_umum": "Jasa_umum.json",
#     "bis_truk": "Bis_Truk.json",
#     "jasa_bis_truk": "Jasa_Bis_Truk.json"
# }

# # Gabungan (target tabel)
# merge_map = {
#     "gabungan_umum": ["Umum.json", "Jasa_umum.json"],
#     "gabungan_bis_truk": ["Bis_Truk.json", "Jasa_Bis_Truk.json"],
#     "qa_service": ["Qa_Service.json"],
#     "oli_fix": ["Oli_Fix.json"]
# }

# harga_file = "harga_data.json"


# # ============================
# # 2. CONNECT DATABASE
# # ============================
# conn = sqlite3.connect(db_path)
# cursor = conn.cursor()

# print("🔌 Terhubung ke database:", db_path)


# # ============================
# # 3. CREATE TABLE SQuAD
# # ============================
# for table in merge_map.keys():
#     cursor.execute(f"""
#         CREATE TABLE IF NOT EXISTS {table} (
#             id TEXT PRIMARY KEY,
#             category TEXT,
#             question TEXT,
#             answer TEXT,
#             context TEXT,
#             keyword TEXT
#         )
#     """)

# # table harga
# cursor.execute("""
#     CREATE TABLE IF NOT EXISTS harga_data (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         placeholder TEXT,
#         harga TEXT,
#         source_file TEXT
#     )
# """)

# conn.commit()


# # ==========================================
# # 4. FUNCTION EXTRACT KEYWORD DARI CONTEXT
# # ==========================================
# def extract_keywords(context):
#     """
#     Mengambil isi dalam [Keyword: ...]
#     contoh:
#     "[Keyword: kapan, baik, ganti ban] Kalimat..." → "kapan, baik, ganti ban"
#     """
#     match = re.search(r"\[Keyword:\s*(.*?)\]", context)
#     if match:
#         return match.group(1).strip()
#     return ""


# # ====================================================
# # 5. FUNCTION IMPORT SQuAD DARI BEBERAPA FILE SEKALIGUS
# # ====================================================
# def import_squad_files(json_files, table_name):
#     for json_file in json_files:
#         path = os.path.join(json_folder, json_file)
#         print(f"📥 Import SQuAD: {json_file}")

#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         for item in data["data"]:
#             category = item.get("title", "")

#             for paragraph in item.get("paragraphs", []):
#                 context = paragraph.get("context", "")

#                 # ambil keyword dari context
#                 keyword = extract_keywords(context)

#                 for qa in paragraph.get("qas", []):
#                     qid = qa.get("id")
#                     question = qa.get("question", "")
#                     answer = qa["answers"][0]["text"]

#                     cursor.execute(f"""
#                         INSERT OR REPLACE INTO {table_name}
#                         (id, category, question, answer, context, keyword)
#                         VALUES (?, ?, ?, ?, ?, ?)
#                     """, (qid, category, question, answer, context, keyword))

#         conn.commit()
#         print(f"✅ Sukses import → {table_name} ({json_file})")


# # ============================
# # 6. JALANKAN IMPORT SQuAD
# # ============================
# for table_name, file_list in merge_map.items():
#     import_squad_files(file_list, table_name)


# # ============================
# # 7. IMPORT harga_data.json
# # ============================
# print("\n📥 Import harga_data.json")

# harga_path = os.path.join(json_folder, harga_file)

# with open(harga_path, "r", encoding="utf-8") as f:
#     harga_data = json.load(f)

# for item in harga_data:
#     placeholder = item.get("placeholder", "")
#     harga = item.get("harga", "")
#     source_file = item.get("source_file", "")

#     cursor.execute("""
#         INSERT INTO harga_data (placeholder, harga, source_file)
#         VALUES (?, ?, ?)
#     """, (placeholder, harga, source_file))

# conn.commit()
# print("💰 Selesai import harga_data!")

# # ============================
# # 8. FINISH
# # ============================
# conn.close()
# print("\n🎉 SEMUA DATA BERHASIL DIMASUKKAN KE bengkel.db!")
