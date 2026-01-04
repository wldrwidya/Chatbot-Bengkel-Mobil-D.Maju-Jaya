import os
import json

folder_path = r"C:\Users\Widya HW\OneDrive - Universitas Tarumanagara\Desktop\Skripsi\Fix Program\converted_jsons\QA_Final"

combined_data = {"data": []}

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

            # Case 1: JSON format SQuAD {"data": [...]}
            if isinstance(json_data, dict) and "data" in json_data:
                combined_data["data"].extend(json_data["data"])

            # Case 2: JSON langsung berupa list
            elif isinstance(json_data, list):
                combined_data["data"].extend(json_data)

            # Case 3: Format aneh → diberi warning
            else:
                print(f"⚠️ Format JSON tidak dikenali: {filename}")

output_path = os.path.join(folder_path, "ALL_QA_combined.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(combined_data, f, indent=2, ensure_ascii=False)

print(f"File gabungan berhasil dibuat: {output_path}")
