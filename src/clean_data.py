import json, re, unicodedata
from urllib.parse import urlparse, parse_qs
from tqdm import tqdm
from collections import defaultdict

# =========================
# 📁 ĐƯỜNG DẪN FILE
# =========================
INPUT_FILE  = "../data/vbpl_data_hoan_chinh.jsonl"
OUTPUT_FILE = "../data/vbpl_data_clean.jsonl"
LOG_FILE    = "../data/clean_log.txt"

# =========================
# 🔤 HÀM CHUẨN HÓA CHUỖI
# =========================
def normalize_unicode(s: str) -> str:
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFC", s)
    return (s.replace("\u00A0", " ")
              .replace("\u200B", "")
              .replace("\ufeff", "")
              .replace("\r", " "))

def clean_space(s: str) -> str:
    if not isinstance(s, str): return ""
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

# =========================
# 🧩 HÀM LẤY ID ỔN ĐỊNH
# =========================
def get_unique_id(item):
    """Tạo ID duy nhất dựa trên ItemID trong URL hoặc fallback theo hash."""
    url = item.get("url_goc", "")
    title = item.get("tieu_de", "")
    attrs = item.get("thuoc_tinh", {}) or {}

    try:
        qs = parse_qs(urlparse(url).query)
        item_id = qs.get("ItemID", [None])[0]
        if item_id:
            return f"vbpl_{item_id}"
    except Exception:
        pass

    sig = f"{title}_{attrs.get('Số ký hiệu','')}_{attrs.get('Ngày ban hành','')}"
    return f"auto_{abs(hash(sig))}"

# =========================
# 🧹 HÀM LÀM SẠCH
# =========================
def clean_item(item):
    """Chuẩn hóa 1 record."""
    item["tieu_de"] = clean_space(normalize_unicode(item.get("tieu_de", "")))
    item["url_goc"] = clean_space(item.get("url_goc", ""))
    item["noi_dung"] = normalize_unicode(item.get("noi_dung", ""))

    if not isinstance(item.get("thuoc_tinh"), dict):
        item["thuoc_tinh"] = {}

    for k, v in list(item["thuoc_tinh"].items()):
        if isinstance(v, str):
            item["thuoc_tinh"][k] = clean_space(normalize_unicode(v))
        else:
            item["thuoc_tinh"][k] = str(v)

    return item

# =========================
# 🚀 MAIN
# =========================
def main():
    seen = {}
    duplicate_groups = defaultdict(list)
    total, kept, skip = 0, 0, 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout, \
         open(LOG_FILE, "w", encoding="utf-8") as flog:

        for line in tqdm(fin, desc="🔍 Cleaning data"):
            total += 1
            line = line.strip()
            if not line: 
                continue

            try:
                item = json.loads(line)
            except Exception as e:
                flog.write(f"Lỗi JSON dòng {total}: {repr(e)}\n")
                skip += 1
                continue

            item = clean_item(item)
            uid = get_unique_id(item)

            # bỏ văn bản không có nội dung
            if len(item.get("noi_dung", "").strip()) < 200:
                flog.write(f"Bỏ văn bản trống: {uid}\n")
                skip += 1
                continue

            # nếu trùng thì giữ bản dài hơn
            if uid in seen:
                old = seen[uid]
                if len(item["noi_dung"]) > len(old["noi_dung"]):
                    duplicate_groups[uid].append(old)
                    seen[uid] = item
                else:
                    duplicate_groups[uid].append(item)
            else:
                seen[uid] = item

        # ghi file kết quả
        for uid, it in seen.items():
            it["unique_id"] = uid
            fout.write(json.dumps(it, ensure_ascii=False) + "\n")
            kept += 1

        # ghi log duplicates
        flog.write("\n=== DUPLICATE SUMMARY ===\n")
        for uid, group in duplicate_groups.items():
            flog.write(f"{uid}: {len(group)} bản trùng\n")

    print("✅ Hoàn tất làm sạch!")
    print(f"• Tổng đọc: {total:,}")
    print(f"• Giữ lại: {kept:,}")
    print(f"• Bỏ qua (trống/lỗi): {skip:,}")
    print(f"• Duplicate nhóm: {len(duplicate_groups):,}")
    print(f"• File sạch: {OUTPUT_FILE}")
    print(f"• Log: {LOG_FILE}")

if __name__ == "__main__":
    main()
