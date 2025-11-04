# VBPL Chatbot

Dự án xây dựng chatbot hỏi đáp về văn bản pháp luật Việt Nam.  
Toàn bộ pipeline chạy cục bộ (local) với mô hình ngôn ngữ Llama 3.2 và cơ sở dữ liệu vector Chroma.

---

## Mục tiêu

- Thu thập và làm sạch dữ liệu từ vbpl.vn  
- Chia văn bản theo cấu trúc Chương, Mục, Điều  
- Tạo embedding tiếng Việt bằng AITeamVN/Vietnamese_Embedding  
- Lưu trữ và truy xuất vector với ChromaDB  
- Thực hiện RAG (Retrieval-Augmented Generation) với Llama 3.2 qua Ollama

---

## Cấu trúc thư mục

chatbot_vbpl/
│
├── src/
│ ├── data_crawl.py # Thu thập danh sách liên kết văn bản
│ ├── data_scrape.py # Cào nội dung và thuộc tính chi tiết
│ ├── clean_data.py # Làm sạch và chuẩn hóa dữ liệu
│ ├── chunking.py # Chia nhỏ văn bản theo Điều / Mục
│ ├── embedding.py # Tạo embedding và lưu vào ChromaDB
│ ├── rag_chatbot.py # Chatbot RAG sử dụng Llama 3.2
│ └── sample1k.py # Tạo tập mẫu 1k để thử nghiệm
│
├── data/ # Lưu trữ file dữ liệu JSONL
├── chroma_db/ # Vector database của Chroma
├── requirements.txt
└── .gitignore
