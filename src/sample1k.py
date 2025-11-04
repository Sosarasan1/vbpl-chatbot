import json
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

input_path = os.path.join(CURRENT_DIR, "..", "data", "vbpl_data_clean.jsonl")
output_path = os.path.join(CURRENT_DIR, "..", "data", "vbpl_data_sample_1k.jsonl")

with open(input_path, "r", encoding="utf-8") as f_in:
    lines = [next(f_in) for _ in range(1000)]

with open(output_path, "w", encoding="utf-8") as f_out:
    f_out.writelines(lines)

print(f"Đã tạo {output_path} với {len(lines)} văn bản (1.000 dòng đầu tiên).")
