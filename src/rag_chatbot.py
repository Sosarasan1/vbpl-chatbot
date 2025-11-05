import json, numpy as np, re, requests, pickle, os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from chromadb import PersistentClient
import tiktoken
from hashlib import sha256

# ===== CONFIG =====
CHROMA_PATH = "/mnt/d/chatbot_vbpl/chroma_db"
CACHE_FILE = "/mnt/d/chatbot_vbpl/query_cache.pkl"

COLLECTION_NAME = "vbpl"
EMBED_MODEL_NAME = "AITeamVN/Vietnamese_Embedding"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
LLM_MODEL = "llama3.2:3b"
TOP_K = 10
CONTEXT_TOKEN_LIMIT = 2500
SIM_THRESHOLD = 0.35   # 🔹 Ngưỡng tương đồng tối thiểu

# ===== LOAD MODEL =====
print("🔹 Đang load model embedding...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cuda")

# ===== CONNECT CHROMA =====
print("🔹 Kết nối ChromaDB...")
client = PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)

# ===== TOKEN LIMIT =====
tokenizer = tiktoken.get_encoding("cl100k_base")
def truncate_context_by_token(context, max_tokens=CONTEXT_TOKEN_LIMIT):
    tokens = tokenizer.encode(context)
    if len(tokens) <= max_tokens:
        return context
    return tokenizer.decode(tokens[:max_tokens]) + "\n...[rút gọn do vượt giới hạn]"

# ===== QUERY NORMALIZATION =====
def normalize_query(q: str) -> str:
    q = q.strip().lower()
    q = re.sub(r"điều\s*(\d+)", r"Điều \1", q, flags=re.I)
    q = re.sub(r"chương\s*([IVXLCDM]+)", r"Chương \1", q, flags=re.I)
    q = re.sub(r"nghị\s*định\s*(\d+)", r"Nghị định \1", q, flags=re.I)
    q = re.sub(r"luật\s*(\d+)", r"Luật \1", q, flags=re.I)
    return q

# ===== CACHE =====
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        query_cache = pickle.load(f)
else:
    query_cache = {}

def get_query_embedding(query: str) -> np.ndarray:
    """Trả về vector embedding, có cache"""
    key = sha256(query.encode("utf-8")).hexdigest()
    if key in query_cache:
        return np.array(query_cache[key])
    emb = embed_model.encode([tokenize(query)], convert_to_numpy=True)[0]
    query_cache[key] = emb
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(query_cache, f)
    return emb

# ===== RERANK COSINE =====
def rerank_cosine(query_emb: np.ndarray, doc_embs: np.ndarray, docs, top_k: int = 5):
    sims = cosine_similarity(np.array([query_emb]), np.array(doc_embs))[0]
    ranked_idx = np.argsort(sims)[::-1][:top_k]
    return [{"text": docs[i]["text"], "meta": docs[i]["meta"], "score": float(sims[i])} for i in ranked_idx]

# ===== MAIN CHAT LOOP =====
print("\n🤖 Chatbot VBPL sẵn sàng. Gõ 'exit' để thoát.\n")

while True:
    query = input("👤 Bạn: ").strip()
    if query.lower() in ["exit", "quit", "q"]:
        print("Tạm biệt nhé 👋")
        break

    norm_query = normalize_query(query)
    q_emb = get_query_embedding(norm_query)

    # --- Truy vấn Chroma ---
    match = re.search(r"điều\s*\d+", norm_query)
    where_filter = {"$contains": match.group(0)} if match else None

    if where_filter:
        results = collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=TOP_K * 2,
            where_document=where_filter,
            include=["documents", "metadatas"]
        )
    else:
        results = collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=TOP_K * 2,
            include=["documents", "metadatas"]
        )

    docs_raw = (results.get("documents") or [[]])[0]
    metas_raw = (results.get("metadatas") or [[]])[0]

    if not docs_raw:
        print("⚠️ Không tìm thấy dữ liệu phù hợp.\n")
        continue

    # --- Rerank cosine ---
    doc_embs = embed_model.encode([tokenize(d) for d in docs_raw], convert_to_numpy=True)
    docs_struct = [{"text": d, "meta": m} for d, m in zip(docs_raw, metas_raw)]
    top_docs = rerank_cosine(q_emb, doc_embs, docs_struct, top_k=5)

    # 🔹 Kiểm tra ngưỡng tương đồng
    avg_sim = np.mean([d["score"] for d in top_docs])
    if avg_sim < SIM_THRESHOLD:
        print("🤔 Câu hỏi này có vẻ không thuộc phạm vi pháp luật Việt Nam.\n"
              "Tôi chỉ cung cấp thông tin dựa trên các văn bản pháp luật được lưu trữ trong cơ sở dữ liệu.\n")
        continue

    # Lọc chỉ giữ đoạn liên quan
    top_docs = [d for d in top_docs if d["score"] >= SIM_THRESHOLD]
    if not top_docs:
        print("⚠️ Không có đoạn nào đủ liên quan để trả lời.\n")
        continue

    print("\n Top 3 đoạn được chọn (sau rerank):")
    for i, d in enumerate(top_docs[:3], 1):
        title = d["meta"].get("title", "Không rõ")
        sec = d["meta"].get("section_path", "")
        print(f"{i}. {title} ({sec}) | cosine={d['score']:.4f}")
    print("──────────────────────────────────────────────\n")

    # --- Context ---
    context = "\n\n".join([
        f"📘 {d['meta'].get('title','')} ({d['meta'].get('section_path','')})\n{d['text']}"
        for d in top_docs[:3]
    ])
    context = truncate_context_by_token(context)

    # --- Prompt ---
    prompt = f"""
📘 NGỮ CẢNH (trích từ các văn bản pháp luật Việt Nam):
{context}

🧩 CÂU HỎI:
{query}

---
Hãy trả lời hoàn toàn bằng **tiếng Việt chuẩn pháp lý**, rõ ràng và chính xác.
Dựa vào NGỮ CẢNH ở trên để **trích dẫn hoặc tóm tắt nội dung liên quan nhất** đến câu hỏi.

Yêu cầu:
- Nếu không có thông tin liên quan, hãy trả lời:
  "Câu hỏi này không nằm trong phạm vi văn bản pháp luật Việt Nam."
- Không bịa đặt hoặc suy luận ngoài nội dung NGỮ CẢNH.
- Giữ giọng văn hành chính, pháp lý, trung lập.
---
"""

    try:
        payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False}
        res = requests.post(OLLAMA_URL, json=payload)
        if res.status_code == 200:
            answer = res.json().get("response", "").strip()
            print(f"\n🧠 {answer}\n")
        else:
            print(f"❌ Lỗi Ollama ({res.status_code}): {res.text}\n")
    except Exception as e:
        print(f"❌ Exception: {e}\n")
