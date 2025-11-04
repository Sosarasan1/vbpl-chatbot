from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from chromadb import PersistentClient
import json
from tqdm import tqdm

# ====== CONFIG ======
MODEL_NAME = "AITeamVN/Vietnamese_Embedding"
INPUT_FILE = "../data/chunks_1k.jsonl"          
CHROMA_PATH = "../chroma_db"                    
COLLECTION_NAME = "vbpl"
BATCH_SIZE = 5000

# ====== LOAD MODEL ======
print(f"📦 Đang load model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, device="cuda")

# ====== LOAD DỮ LIỆU ======
print(f"📖 Đọc dữ liệu từ {INPUT_FILE} ...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

texts = [d["text"] for d in data]
print(f"🔹 Tổng số chunks: {len(texts):,}")

# ====== TOKENIZE ======
print("🔤 Tokenizing tiếng Việt ...")
tokenized = [tokenize(t) for t in tqdm(texts, desc="Tokenizing")]

# ====== EMBED ======
print("⚙️ Đang tạo embeddings bằng GPU ...")
embeddings = model.encode(
    tokenized,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

# ====== LƯU THEO BATCH VÀO CHROMA ======
print(f"💾 Lưu vào ChromaDB: {CHROMA_PATH}")
client = PersistentClient(path=CHROMA_PATH)

# Nếu collection đã tồn tại → lấy lại thay vì tạo mới
try:
    collection = client.get_collection(COLLECTION_NAME)
    print(f"⚠️ Collection '{COLLECTION_NAME}' đã tồn tại, ghi thêm dữ liệu.")
except:
    collection = client.create_collection(COLLECTION_NAME)
    print(f"🆕 Tạo collection mới: {COLLECTION_NAME}")

# Ghi theo batch an toàn
for i in range(0, len(data), BATCH_SIZE):
    batch_data = data[i:i+BATCH_SIZE]
    batch_emb = embeddings[i:i+BATCH_SIZE].tolist()

    collection.add(
        ids=[d["item_id"] for d in batch_data],
        embeddings=batch_emb,
        metadatas=[{
            "title": d["title"],
            "section_path": d.get("section_path", "")
        } for d in batch_data],
        documents=[d["text"] for d in batch_data]
    )

    print(f"✅ Batch {i//BATCH_SIZE + 1}: đã thêm {len(batch_data)} chunks")

print(f"\n🎉 Hoàn tất! Tổng cộng {len(data):,} chunks đã lưu vào '{COLLECTION_NAME}' trong '{CHROMA_PATH}'")

# ====== KIỂM TRA LẠI SỐ LƯỢNG ======
print(f"📊 Tổng vectors hiện có trong collection: {collection.count()}")
