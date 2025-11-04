import re, json, unicodedata, hashlib, os
from urllib.parse import urlparse, parse_qs
from tqdm import tqdm
import tiktoken

# ===== FILE CONFIG =====
INPUT_FILE  = "../data/vbpl_data_sample_1k.jsonl"   # dùng file 1k đã chọn
OUTPUT_FILE = "../data/chunks_1k.jsonl"             # file chunk mới sinh ra
LOG_FILE    = "../data/merged_short_chunks_1k.txt"

# ===== CHUNK LIMITS =====
MAX_CHARS   = 1800
MIN_CHARS   = 200
MAX_TOKENS  = 450
MIN_TOKENS  = 10

tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def normalize_unicode(s: str) -> str:
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFC", s)
    return s.replace("\u00A0", " ").replace("\u200B", "").replace("\ufeff", "").replace("\r", " ")

def clean_field(s: str) -> str:
    if not isinstance(s, str): return ""
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

RE_CHUONG = re.compile(r"^\s*Chương\s+[IVXLCDM]+\b.*$", re.MULTILINE)
RE_MUC    = re.compile(r"^\s*Mục\s+\d+\b.*$", re.MULTILINE)
RE_DIEU   = re.compile(r"^\s*(Điều\s+\d+)\b", re.MULTILINE)

def find_last(regex, text, pos):
    matches = [m.group(0).strip() for m in regex.finditer(text[:pos])]
    return matches[-1] if matches else None

def split_by_dieu(full_text: str):
    text = normalize_unicode(full_text)
    matches = list(RE_DIEU.finditer(text))
    if not matches:
        return [(None, text.strip())]
    blocks = []
    for i, m in enumerate(matches):
        start = m.start(1)
        end   = matches[i+1].start(1) if i+1 < len(matches) else len(text)
        block = text[start:end].strip()
        chuong = find_last(RE_CHUONG, text, start)
        muc    = find_last(RE_MUC, text, start)
        dieu   = m.group(1).strip()
        section_path = " > ".join([p for p in [chuong, muc, dieu] if p])
        blocks.append((section_path, block))
    return blocks

# --- NGẮT CÂU: fix lỗi liệt kê dài ---
SENT_BOUNDARY = re.compile(r"(?<=[\.\?!;,:])\s+|\n+")

def split_into_sentences(text: str):
    """Tách câu mềm, fallback nếu chỉ có 1 đoạn quá dài."""
    parts = [p.strip() for p in re.split(SENT_BOUNDARY, text) if p.strip()]
    if len(parts) == 1 and len(parts[0]) > 1000:
        subparts = [parts[0][i:i+800] for i in range(0, len(parts[0]), 800)]
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[FALLBACK SPLIT] Đoạn dài {len(parts[0])} ký tự, chia cưỡng bức thành {len(subparts)} phần\n")
        return subparts
    return parts if parts else [text]

def pack_sentences(sentences, max_chars=MAX_CHARS, max_tokens=MAX_TOKENS, min_tokens=MIN_TOKENS):
    """Gộp câu thành chunk ≤ max_chars & max_tokens, gộp chunk ngắn."""
    chunks, buf = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) > max_chars:
            if buf:
                chunks.append(buf.strip())
                buf = ""
            start = 0
            while start < len(s):
                end = min(start + max_chars, len(s))
                space_idx = s.rfind(" ", start, end)
                if space_idx == -1 or space_idx - start < 30:
                    space_idx = end
                sub = s[start:space_idx].strip()
                if sub:
                    chunks.append(sub)
                start = space_idx
            continue
        if count_tokens(buf + " " + s) <= max_tokens and len(buf) + len(s) <= max_chars:
            buf = (buf + " " + s) if buf else s
        else:
            if buf:
                chunks.append(buf.strip())
            buf = s
    if buf:
        chunks.append(buf.strip())

    # Gộp các chunk ngắn
    merged, i = [], 0
    while i < len(chunks):
        curr = chunks[i].strip()
        tok = count_tokens(curr)
        if tok < min_tokens and merged:
            merged[-1] = (merged[-1] + " " + curr).strip()
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[MERGE BACKWARD] Gộp chunk ngắn {tok} tokens: {curr[:100]}\n")
        elif tok < min_tokens and i + 1 < len(chunks):
            nxt = chunks[i + 1].strip()
            merged.append((curr + " " + nxt).strip())
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[MERGE FORWARD] Gộp chunk ngắn {tok} tokens: {curr[:100]}\n")
            i += 1
        else:
            merged.append(curr)
        i += 1
    return merged

def extract_doc_id(url: str, title: str, attrs: dict) -> str:
    try:
        qs = parse_qs(urlparse(url).query)
        item_id = qs.get("ItemID", [None])[0]
        if item_id:
            return f"vbpl_{item_id}"
    except:
        pass
    raw = f"{url}|{title}|{attrs.get('Ngày ban hành', '')}"
    return "doc_" + hashlib.md5(raw.encode("utf-8")).hexdigest()

def process_item(item):
    title = clean_field(item.get("tieu_de", ""))
    url   = item.get("url_goc", "")
    attrs = item.get("thuoc_tinh", {}) or {}
    for k, v in list(attrs.items()):
        if isinstance(v, str):
            attrs[k] = clean_field(v)
    full_text = normalize_unicode(item.get("noi_dung", ""))
    if not full_text.strip():
        return []

    doc_id = extract_doc_id(url, title, attrs)
    dieu_blocks = split_by_dieu(full_text)
    chunks = []

    for chunk_index, (section_path, block) in enumerate(dieu_blocks):
        section_path = clean_field(section_path or "")
        sentences = split_into_sentences(block)
        parts = pack_sentences(sentences)
        for part_index, sub in enumerate(parts):
            tokens = count_tokens(sub)
            chunk_id = hashlib.md5(f"{doc_id}|{section_path}|{chunk_index}|{part_index}".encode()).hexdigest()
            chunks.append({
                "item_id": chunk_id,
                "doc_id": doc_id,
                "title": title,
                "url": url,
                "attrs": attrs,
                "section_path": section_path,
                "text": sub,
                "tokens": tokens
            })
    return chunks


# ===== MAIN PIPELINE =====
total_chunks = 0
total_tokens = 0
min_chunk = {"tokens": float("inf"), "text": ""}
max_chunk = {"tokens": 0, "text": ""}

if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

with open(INPUT_FILE, "r", encoding="utf-8") as fin, open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for line in tqdm(fin, desc="Chunking từ vbpl_data_sample_1k"):
        if not line.strip():
            continue
        obj = json.loads(line)
        result = process_item(obj)
        for ch in result:
            fout.write(json.dumps(ch, ensure_ascii=False) + "\n")
            total_chunks += 1
            total_tokens += ch["tokens"]

            if ch["tokens"] < min_chunk["tokens"]:
                min_chunk = {"tokens": ch["tokens"], "text": ch["text"][:200]}
            if ch["tokens"] > max_chunk["tokens"]:
                max_chunk = {"tokens": ch["tokens"], "text": ch["text"][:200]}

print("\n✅ Hoàn tất chunking 1k VBPL")
print(f"📦 File đầu ra: {OUTPUT_FILE}")
print(f"🔢 Tổng chunk: {total_chunks}")
print(f"🔠 Tổng tokens: {total_tokens}")
print(f"⚖️ Trung bình tokens/chunk: {total_tokens // max(1, total_chunks)}")
print(f"⬇️ Chunk nhỏ nhất: {min_chunk['tokens']} tokens → {min_chunk['text']}")
print(f"⬆️ Chunk lớn nhất: {max_chunk['tokens']} tokens → {max_chunk['text']}")
print(f"🧾 Log chi tiết tại: {LOG_FILE}")
