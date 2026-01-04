import json  # untuk membaca file data berformat JSON
import numpy as np  # untuk operasi data numerik, walau dipakai minimal

from datasets import Dataset, DatasetDict  # untuk membuat dataset HuggingFace
from transformers import (
    AutoTokenizer,                    # tokenizer untuk ubah teks jadi token
    AutoModelForQuestionAnswering,    # model IndoBERT khusus QA
    TrainingArguments,                # parameter training
    Trainer,                          # class untuk proses training otomatis
    default_data_collator,            # untuk batching data ke model
)
from evaluate import load  # untuk menghitung metric evaluasi (SQuAD)

# === 1. LOAD DATASET FORMAT SQuAD ===
with open(r"C:\Users\Widya HW\OneDrive - Universitas Tarumanagara\Desktop\Skripsi\Fix Program\converted_jsons\QA_Final\all_qa\ALL_QA_combined.json", "r", encoding="utf-8") as f:
    squad_data = json.load(f)  # buka dan baca seluruh file JSON dataset

# Konversi JSON ke list of dict agar bisa dibuat Dataset
samples = []  # tempat menampung daftar sample QA
for entry in squad_data["data"]:  # loop tiap kategori data
    for para in entry["paragraphs"]:  # loop tiap paragraf
        context = para["context"]  # ambil teks paragraf
        for qa in para["qas"]:  # loop tiap QA di paragraf
            if qa["answers"]:  # pastikan ada jawaban
                samples.append({
                    "id": qa["id"],               # simpan id soal
                    "question": qa["question"],   # simpan pertanyaan
                    "context": context,           # simpan teks konteks
                    "answers": qa["answers"][0]   # simpan jawaban pertama
                })

# Buat Dataset HuggingFace (train/validation)
dataset = Dataset.from_list(samples)  # ubah list jadi Dataset HF
dataset = dataset.train_test_split(test_size=0.2, seed=42)  # split 80/20
dataset = DatasetDict({
    "train": dataset["train"],        # dataset training
    "validation": dataset["test"]     # dataset validation
})

# === 2. TOKENIZER ===
model_name = "cahya/bert-base-indonesian-1.5G"  # nama model IndoBERT
tokenizer = AutoTokenizer.from_pretrained(model_name)  # load tokenizer IndoBERT

def prepare_train_features(examples):  # fungsi untuk tokenisasi + mapping jawaban
    tokenized = tokenizer(
        examples["question"],         # input pertanyaan
        examples["context"],          # input konteks
        truncation="only_second",     # potong teks jika panjang
        max_length=384,               # panjang token maksimal
        stride=128,                   # overlap sliding window
        return_overflowing_tokens=True, # memungkinkan split panjang
        return_offsets_mapping=True,  # simpan posisi karakter asli
        padding="max_length"          # padding ke max token size
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")  # map potongan teks
    offset_mapping = tokenized.pop("offset_mapping")  # boundary karakter tiap token

    start_positions = []  # list posisi start token jawaban
    end_positions = []    # list posisi end token jawaban

    for i, offsets in enumerate(offset_mapping):  # loop tiap offset token
        input_ids = tokenized["input_ids"][i]  # ambil token id
        cls_index = input_ids.index(tokenizer.cls_token_id)  # posisi token CLS
        sequence_ids = tokenized.sequence_ids(i)  # info token context/question
        sample_index = sample_mapping[i]  # ambil index sample asli
        answers = examples["answers"][sample_index]  # ambil jawaban asli
        start_char = answers["answer_start"]  # posisi mulai jawaban di teks
        end_char = answers.get("answer_end", start_char + len(answers["text"]))  # posisi akhir char
        context_index = 1  # context = segment 1

        token_start_index = 0  # cari awal token konteks
        while sequence_ids[token_start_index] != context_index:
            token_start_index += 1

        token_end_index = len(input_ids) - 1  # cari akhir token konteks
        while sequence_ids[token_end_index] != context_index:
            token_end_index -= 1

        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            start_positions.append(cls_index)  # tidak ditemukan → CLS
            end_positions.append(cls_index)
        else:
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            start_positions.append(token_start_index - 1)  # simpan index token start

            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)  # simpan index token end

    tokenized["start_positions"] = start_positions  # tambahkan posisi start
    tokenized["end_positions"] = end_positions      # tambahkan posisi end
    return tokenized

tokenized_datasets = dataset.map(
    prepare_train_features,   # fungsi tokenisasi custom
    batched=True,             # proses batch
    remove_columns=dataset["train"].column_names  # hapus kolom asli
)

# === 3. LOAD MODEL ===
model = AutoModelForQuestionAnswering.from_pretrained(model_name)  # load IndoBERT QA

# === 4. TRAINING CONFIG ===
training_args = TrainingArguments(
    output_dir="./results_qa_squad_final_V2",  # folder output
    evaluation_strategy="epoch",            # evaluasi tiap epoch
    save_strategy="epoch",                 # save tiap epoch
    learning_rate=3e-5,                    # learning rate
    per_device_train_batch_size=8,         # batch size train
    per_device_eval_batch_size=8,          # batch size eval
    num_train_epochs=3,                    # epoch total
    weight_decay=0.01,                     # regularization
    logging_dir="./logs_qa_final",         # folder log
    logging_strategy="epoch",             # log tiap epoch
    report_to="none",                     # tidak upload ke wandb/tb
)

# # === 5. METRICS (Opsional) ===
# metric = load("squad")  # load evaluasi SQuAD metric

# def compute_metrics(p):  # fungsi hitung EM + F1
#     predictions, references = p
#     return metric.compute(predictions=predictions, references=references)

# === 6. TRAINER ===
trainer = Trainer(
    model=model,                             # model IndoBERT
    args=training_args,                      # config training
    train_dataset=tokenized_datasets["train"],  # data training
    eval_dataset=tokenized_datasets["validation"], # data validasi
    tokenizer=tokenizer,                     # tokenizer
    data_collator=default_data_collator,     # collator default
)
# === CEK JUMLAH DATASET ===
print("Jumlah sample asli (train):", len(dataset["train"]))
print("Jumlah sample asli (validation):", len(dataset["validation"]))

print("Jumlah chunk tokenized (train):", tokenized_datasets["train"].shape)
print("Jumlah chunk tokenized (validation):", tokenized_datasets["validation"].shape)

# === 7. TRAINING ===
train_result = trainer.train()  # mulai training model

# === 8. SAVE MODEL ===
save_path = "indobert-qa-finetuned-final-V2"  # nama folder model akhir
model.save_pretrained(save_path)  # simpan model hasil training
tokenizer.save_pretrained(save_path)  # simpan tokenizer

print(f"✅ Fine-tuning selesai. Model disimpan di folder '{save_path}'") # info selesai
