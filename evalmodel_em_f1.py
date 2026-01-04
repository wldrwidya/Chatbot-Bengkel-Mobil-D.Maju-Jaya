import json
import re
import os

# === Folder tempat file berada ===
BASE_DIR = r"C:\Users\Widya HW\OneDrive - Universitas Tarumanagara\Desktop\Skripsi\Fix Program\converted_jsons\QA_V2"

# === Normalisasi teks ===
def normalize_text(s):
    s = s.lower().strip()
    # hilangkan tanda baca umum
    s = re.sub(r'[^\w\s]', ' ', s)
    # samakan format angka: ubah "120rb" jadi "120000"
    s = re.sub(r'(\d+)\s?rb', lambda m: str(int(m.group(1)) * 1000), s)
    # hapus "rp" di depan angka
    s = re.sub(r'rp\s?', '', s)
    # ubah titik jadi kosong (untuk angka kayak 120.000)
    s = s.replace('.', '')
    # ubah banyak spasi ke satu spasi
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

# === Hitung Exact Match ===
def compute_exact_match(prediction, ground_truth):
    return int(normalize_text(prediction) == normalize_text(ground_truth))

# === Hitung Soft Exact Match (kemiripan ≥ 90%) ===
def compute_soft_em(prediction, ground_truth, threshold=0.6):
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0

    common = set(pred_tokens) & set(gt_tokens)
    overlap = (2 * len(common)) / (len(pred_tokens) + len(gt_tokens))
    return int(overlap >= threshold)

# === Hitung F1 Score ===
def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    common = set(pred_tokens) & set(gt_tokens)
    if not pred_tokens or not gt_tokens:
        return 0.0
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# === Evaluasi satu pasangan file ===
def evaluate(pred_file, dataset_file):
    pred_path = os.path.join(BASE_DIR, pred_file)
    data_path = os.path.join(BASE_DIR, dataset_file)

    with open(pred_path, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_em, total_soft_em, total_f1, n = 0, 0, 0, 0

    for item in preds:
        gt = item.get("ground_truth", "")
        pred = item.get("prediction", "")
        total_em += compute_exact_match(pred, gt)
        total_soft_em += compute_soft_em(pred, gt)
        total_f1 += compute_f1(pred, gt)
        n += 1

    em = total_em / n * 100 if n else 0
    soft_em = total_soft_em / n * 100 if n else 0
    f1 = total_f1 / n * 100 if n else 0

    return f"📊 {pred_file}\n  - Total Pertanyaan: {n}\n  - Exact Match: {em:.2f}%\n  - Soft Exact Match: {soft_em:.2f}%\n  - F1 Score: {f1:.2f}%\n\n"

# === Jalankan evaluasi semua file ===
results = ""
results += evaluate("predictions_qa_service.json", "qa_service.json")
results += evaluate("predictions_oli_fix.json", "oli_fix.json")
results += evaluate("predictions_umum.json", "gabungan_umum.json")
results += evaluate("predictions_bis_truk.json", "gabungan_bis_truk.json")

# === Simpan hasil ke file txt ===
output_path = os.path.join(BASE_DIR, "evaluation_results.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(results)

print("✅ Evaluasi selesai! Hasil disimpan di:", output_path)
print(results)
