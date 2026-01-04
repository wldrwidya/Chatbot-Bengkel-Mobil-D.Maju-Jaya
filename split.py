import json
import random

# === 1. Load file gabungan ===
with open(
    r"C:\Users\Widya HW\OneDrive - Universitas Tarumanagara\Desktop\Skripsi\Fix Program\converted_jsons\QA_Final\all_qa\ALL_QA_combined.json",
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)

# === 2. Flatten semua QA ke dalam satu list ===
all_qas = []
for entry in data["data"]:
    title = entry.get("title", "Unknown")
    for para in entry["paragraphs"]:
        context = para["context"]
        for qa in para["qas"]:
            new_entry = {
                "title": title,
                "paragraphs": [
                    {
                        "context": context,
                        "qas": [qa]
                    }
                ]
            }
            all_qas.append(new_entry)

# === 3. Acak data dan split train/val ===
random.seed(42)
random.shuffle(all_qas)

split_ratio = 0.8
split_idx = int(len(all_qas) * split_ratio)

train_data = {"data": all_qas[:split_idx]}
val_data = {"data": all_qas[split_idx:]}

# === 4. Simpan ke file ===
with open("train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with open("val.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f, indent=2, ensure_ascii=False)

print(f"✅ Train & Validation dataset berhasil dibuat!")
print(f"Total QA: {len(all_qas)} | Train: {len(train_data['data'])} | Val: {len(val_data['data'])}")
