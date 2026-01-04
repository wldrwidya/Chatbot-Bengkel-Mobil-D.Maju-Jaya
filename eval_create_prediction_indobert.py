import json
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import os

# ========== CONFIG ==========
finetuned_model_path = r"C:\Users\Widya HW\OneDrive - Universitas Tarumanagara\Desktop\Skripsi\Fix Program\indobert-qa-finetuned-final"

# path dataset
datasets = {
    "qa_service": r"C:\Users\Widya HW\OneDrive - Universitas Tarumanagara\Desktop\Skripsi\Fix Program\converted_jsons\QA_V2\QA_Service.json",
    "oli_fix": r"C:\Users\Widya HW\OneDrive - Universitas Tarumanagara\Desktop\Skripsi\Fix Program\converted_jsons\QA_V2\oli_fix.json",
    "umum": r"C:\Users\Widya HW\OneDrive - Universitas Tarumanagara\Desktop\Skripsi\Fix Program\converted_jsons\QA_V2\gabungan_umum.json",
    "bis_truk": r"C:\Users\Widya HW\OneDrive - Universitas Tarumanagara\Desktop\Skripsi\Fix Program\converted_jsons\QA_V2\gabungan_bis_truk.json",
}
# ============================

# === Load IndoBERT model ===
print("🔹 Loading IndoBERT model...")
tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
model = AutoModelForQuestionAnswering.from_pretrained(finetuned_model_path)
model.eval()

# === Function untuk load dataset ===
def load_dataset(path):
    if not os.path.exists(path):
        print(f"[WARN] File tidak ditemukan: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    qas = []
    for d in data["data"]:
        for p in d["paragraphs"]:
            context = p.get("context", "")
            for q in p.get("qas", []):
                qas.append({
                    "question": q.get("question", ""),
                    "answer": q.get("answers", [{}])[0].get("text", ""),
                    "context": context
                })
    return qas

# === Function prediksi dengan IndoBERT ===
def predict_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1
    tokens = inputs["input_ids"][0][start_idx:end_idx]
    return tokenizer.decode(tokens, skip_special_tokens=True).strip()

# === Generate predictions untuk tiap dataset ===
for name, path in datasets.items():
    print(f"\n🚀 Memproses dataset: {name}")
    qas = load_dataset(path)
    predictions = []

    for i, item in enumerate(qas):
        q = item["question"]
        c = item["context"]
        gt = item["answer"]
        pred = predict_answer(q, c)

        predictions.append({
            "question": q,
            "context": c,
            "ground_truth": gt,
            "prediction": pred
        })

        if (i + 1) % 10 == 0:
            print(f"  > {i + 1}/{len(qas)} pertanyaan selesai...")

    # Simpan hasil prediksi ke file JSON
    output_path = f"predictions_{name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"✅ Hasil disimpan ke: {output_path}")

print("\n🎉 Semua dataset selesai diproses dan disimpan!")
