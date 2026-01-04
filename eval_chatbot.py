# eval_chatbot.py
import os
import json
import re
from collections import Counter, defaultdict

# ---------- CONFIG ----------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # fix_skripsi/
CONVERTED_DIR = os.path.join(BASE_DIR, "converted_jsons")

CHATLOG_FILES = {
    "qa_service": os.path.join(BASE_DIR, "chat_log_service.txt"),
    "oli_fix": os.path.join(BASE_DIR, "chat_log_oli.txt"),
    "gabungan_umum": os.path.join(BASE_DIR, "chat_log_umum_mobil.txt"),
    "gabungan_bis_truk": os.path.join(BASE_DIR, "chat_log_bis_truk.txt"),
}

DATASET_FILES = {
    "qa_service": os.path.join(CONVERTED_DIR, "QA_Service.json"),
    "oli_fix": os.path.join(CONVERTED_DIR, "Oli_Fix.json"),
    "gabungan_umum": os.path.join(CONVERTED_DIR, "gabungan_umum.json"),
    "gabungan_bis_truk": os.path.join(CONVERTED_DIR, "gabungan_bis_truk.json"),
}

OUTPUT_FILES = {
    "qa_service": os.path.join(BASE_DIR, "EVAL_QA_SERVICE.txt"),
    "oli_fix": os.path.join(BASE_DIR, "EVAL_OLI.txt"),
    "gabungan_umum": os.path.join(BASE_DIR, "EVAL_UMUM.txt"),
    "gabungan_bis_truk": os.path.join(BASE_DIR, "EVAL_BIS_TRUK.txt"),
}

# scenario ranges (1-indexed inclusive) per your specification
SCENARIOS = {
    "qa_service": [(1,19), (20,38), (39,48), (49,58), (59,73)],
    "oli_fix": [(1,14), (15,28), (29,38), (39,48), (49,63)],
    "gabungan_umum": [(1,20), (21,30), (31,40), (41,50), (51,65)],
    "gabungan_bis_truk": [(1,14), (15,28), (29,38), (39,48), (49,63)],
}

SOFT_EM_THRESHOLD = 0.75  # token-F1 threshold for soft exact match

# ---------- Helpers ----------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)  # remove punctuation -> spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def exact_match(pred: str, gt: str) -> int:
    return 1 if normalize_text(pred) == normalize_text(gt) and normalize_text(gt) != "" else 0

def token_f1(pred: str, gt: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gt).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    cp = Counter(p)
    cg = Counter(g)
    common = sum(min(cp[t], cg[t]) for t in cp.keys() & cg.keys())
    if common == 0:
        return 0.0
    precision = common / sum(cp.values())
    recall = common / sum(cg.values())
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

def soft_em(pred: str, gt: str, threshold=SOFT_EM_THRESHOLD) -> int:
    return 1 if token_f1(pred, gt) >= threshold else 0

# build map question_normalized -> list of ground truths
def load_dataset_qas(dataset_path):
    mapping = defaultdict(list)
    if not os.path.exists(dataset_path):
        print(f"[WARN] dataset not found: {dataset_path}")
        return mapping
    with open(dataset_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    for item in js.get("data", []):
        for para in item.get("paragraphs", []):
            for qa in para.get("qas", []):
                q_text = qa.get("question", "")
                q_norm = normalize_text(q_text)
                # answers may be list; take all possible ground truth texts
                for ans in qa.get("answers", []):
                    txt = ans.get("text", "")
                    mapping[q_norm].append(txt)
    return mapping

def read_chatlog_lines(path):
    lines = []
    if not os.path.exists(path):
        print(f"[WARN] chatlog not found: {path}")
        return lines
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                lines.append(obj)
            except Exception as e:
                # try to be forgiving: sometimes log lines are plain dict string; skip if invalid
                print(f"[WARN] invalid json in {path} line {i}: {e}")
    return lines

# ---------- Evaluation flow ----------
def evaluate_mode(mode_key):
    chatlog_path = CHATLOG_FILES[mode_key]
    dataset_path = DATASET_FILES[mode_key]
    out_path = OUTPUT_FILES[mode_key]
    scenario_ranges = SCENARIOS[mode_key]

    prints = []
    qmap = load_dataset_qas(dataset_path)  # normalized question -> list of answers
    chat_lines = read_chatlog_lines(chatlog_path)

    # if chatlog empty -> nothing to do but create empty file
    if not chat_lines:
        print(f"[INFO] no chatlog entries for mode {mode_key}")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"No chatlog entries found for mode {mode_key}\n")
        return

    # overall accumulators
    mode_counts = 0
    mode_sum_em = 0
    mode_sum_soft = 0
    mode_sum_f1 = 0.0
    mode_sum_tfidf = 0.0

    # We'll build per-scenario results
    scenario_results = []

    # convert scenario ranges to 0-based slices (inclusive)
    for (start1, end1) in scenario_ranges:
        start_idx = max(0, start1 - 1)
        end_idx = min(len(chat_lines) - 1, end1 - 1)
        # if start_idx > end_idx => no entries
        entries = chat_lines[start_idx:end_idx+1] if start_idx <= end_idx else []
        scenario_results.append({
            "range": (start1, end1),
            "entries": entries,
        })

    # Evaluate each scenario
    out_lines = []
    for si, scen in enumerate(scenario_results, start=1):
        entries = scen["entries"]
        scen_count = 0
        scen_sum_em = 0
        scen_sum_soft = 0
        scen_sum_f1 = 0.0
        scen_sum_tfidf = 0.0
        out_lines.append(f"=== SKENARIO {si} | baris {scen['range'][0]}-{scen['range'][1]} ===\n")
        if not entries:
            out_lines.append(" (tidak ada entri untuk rentang ini)\n\n")
            continue

        for idx, item in enumerate(entries, start=scen['range'][0]):
            # expected structure from your log lines:
            # { "mode": "...", "question": "...", "prediction": "...", "keyword": "...", "tfidf_score": 0.88, ... }
            question = item.get("question", "")
            prediction = item.get("prediction", "") or ""
            keyword = item.get("keyword", "")
            tfidf_score = item.get("tfidf_score", None)
            tfidf_threshold = item.get("tfidf_threshold", None)
            best_index = item.get("best_index", None)
            context_used = item.get("context_used", None)

            qnorm = normalize_text(question)
            gts = qmap.get(qnorm, [])

            # if multiple ground truths, choose best matching GT by highest token F1 against prediction
            chosen_gt = ""
            status = "not_found"
            em_val = 0
            soft_val = 0
            f1_val = 0.0

            if gts:
                status = "found"
                # pick the GT that maximizes token_f1 with prediction (gives best match)
                best_gt = None
                best_gt_f1 = -1.0
                for gt in gts:
                    f1tmp = token_f1(prediction, gt)
                    if f1tmp > best_gt_f1:
                        best_gt_f1 = f1tmp
                        best_gt = gt
                chosen_gt = best_gt or ""
                em_val = exact_match(prediction, chosen_gt)
                f1_val = token_f1(prediction, chosen_gt)
                soft_val = 1 if f1_val >= SOFT_EM_THRESHOLD else 0
            else:
                # not found in dataset: we still compute token_f1 against empty => 0
                status = "not_found"
                chosen_gt = ""
                em_val = 0
                f1_val = 0.0
                soft_val = 0

            # accumulate
            scen_count += 1
            scen_sum_em += em_val
            scen_sum_soft += soft_val
            scen_sum_f1 += f1_val
            if isinstance(tfidf_score, (int, float)):
                scen_sum_tfidf += float(tfidf_score)
            else:
                # try to parse if tfidf_score stored as string
                try:
                    scen_sum_tfidf += float(str(tfidf_score))
                except Exception:
                    pass

            # write per-question JSON-like block
            entry_out = {
                "index_in_chatlog": idx,
                "question": question,
                "prediction": prediction,
                "ground_truth": chosen_gt,
                "status": status,
                "keyword": keyword,
                "tfidf_score": tfidf_score,
                "tfidf_threshold": tfidf_threshold,
                "best_index": best_index,
                "context_used": context_used,
                "em": em_val,
                "soft_em": soft_val,
                "f1": round(f1_val, 4)
            }
            out_lines.append(json.dumps(entry_out, ensure_ascii=False) + "\n")

        # scenario summary
        scen_em_pct = (scen_sum_em / scen_count * 100) if scen_count else 0.0
        scen_soft_pct = (scen_sum_soft / scen_count * 100) if scen_count else 0.0
        scen_f1_avg = (scen_sum_f1 / scen_count * 100) if scen_count else 0.0
        scen_tfidf_avg = (scen_sum_tfidf / scen_count) if scen_count else 0.0

        out_lines.append("\n")
        out_lines.append(f"--- Rata-rata SKENARIO {si} ---\n")
        out_lines.append(f"Jumlah pertanyaan: {scen_count}\n")
        out_lines.append(f"Exact Match (EM): {scen_em_pct:.2f}%\n")
        out_lines.append(f"Soft EM (F1 >= {SOFT_EM_THRESHOLD}): {scen_soft_pct:.2f}%\n")
        out_lines.append(f"F1 (avg *100): {scen_f1_avg:.2f}%\n")
        out_lines.append(f"Rata-rata TF-IDF score: {scen_tfidf_avg:.4f}\n")
        out_lines.append("\n\n")

        # accumulate into overall mode
        mode_counts += scen_count
        mode_sum_em += scen_sum_em
        mode_sum_soft += scen_sum_soft
        mode_sum_f1 += scen_sum_f1
        mode_sum_tfidf += scen_sum_tfidf

    # After all scenarios: overall mode averages
    overall_em_pct = (mode_sum_em / mode_counts * 100) if mode_counts else 0.0
    overall_soft_pct = (mode_sum_soft / mode_counts * 100) if mode_counts else 0.0
    overall_f1_avg = (mode_sum_f1 / mode_counts * 100) if mode_counts else 0.0
    overall_tfidf_avg = (mode_sum_tfidf / mode_counts) if mode_counts else 0.0

    # Build final output file
    header = []
    header.append(f"=== EVALUASI MODE: {mode_key} ===\n")
    header.append(f"Dataset ground truth: {dataset_path}\n")
    header.append(f"Chatlog source: {chatlog_path}\n")
    header.append(f"Jumlah total chatlog entries processed: {mode_counts}\n")
    header.append(f"Soft EM threshold (token-F1): {SOFT_EM_THRESHOLD}\n")
    header.append("\n\n")
    final_text = "".join(header) + "".join(out_lines)
    final_text += "=== RINGKASAN AKHIR MODE ===\n"
    final_text += f"Total pertanyaan (semua skenario): {mode_counts}\n"
    final_text += f"Exact Match (EM): {overall_em_pct:.2f}%\n"
    final_text += f"Soft EM (F1 >= {SOFT_EM_THRESHOLD}): {overall_soft_pct:.2f}%\n"
    final_text += f"F1 (avg *100): {overall_f1_avg:.2f}%\n"
    final_text += f"Rata-rata TF-IDF score: {overall_tfidf_avg:.4f}\n"

    # Additionally print per-scenario summary table
    final_text += "\n=== RINGKASAN PER SKENARIO ===\n"
    # compute again per scenario quickly for table:
    for si, scen in enumerate(scenario_results, start=1):
        entries = scen["entries"]
        if not entries:
            final_text += f"Skenario {si} ({scen['range'][0]}-{scen['range'][1]}): kosong\n"
            continue
        sc_count = len(entries)
        sc_em = sc_soft = sc_f1_sum = sc_tfidf_sum = 0.0
        for item in entries:
            question = item.get("question","")
            prediction = item.get("prediction","") or ""
            qnorm = normalize_text(question)
            gts = qmap.get(qnorm, [])
            if gts:
                # choose best gt by f1
                best_gt = max(gts, key=lambda g: token_f1(prediction, g))
                sc_em += exact_match(prediction, best_gt)
                f1v = token_f1(prediction, best_gt)
                sc_f1_sum += f1v
                sc_soft += 1 if f1v >= SOFT_EM_THRESHOLD else 0
            else:
                sc_em += 0
                sc_soft += 0
                sc_f1_sum += 0.0
            tfidf_score = item.get("tfidf_score", None)
            try:
                sc_tfidf_sum += float(tfidf_score)
            except Exception:
                pass
        sc_em_pct = sc_em / sc_count * 100
        sc_soft_pct = sc_soft / sc_count * 100
        sc_f1_avg = sc_f1_sum / sc_count * 100
        sc_tfidf_avg = sc_tfidf_sum / sc_count
        final_text += (f"Skenario {si} ({scen['range'][0]}-{scen['range'][1]}): "
                       f"Count={sc_count} | EM={sc_em_pct:.2f}% | SoftEM={sc_soft_pct:.2f}% | F1(avg*100)={sc_f1_avg:.2f}% | TFIDF(avg)={sc_tfidf_avg:.4f}\n")

    # write output file
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"[DONE] Evaluasi untuk mode '{mode_key}' ditulis ke: {out_path}")


def main():
    print("Mulai evaluasi chatlog -> dataset ...")
    for key in ["qa_service", "oli_fix", "gabungan_umum", "gabungan_bis_truk"]:
        evaluate_mode(key)
    print("Selesai semua evaluasi.")

if __name__ == "__main__":
    main()
