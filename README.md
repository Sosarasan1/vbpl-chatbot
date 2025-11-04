# VBPL Chatbot

Dự án này được mình phát triển để tìm hiểu cách xây dựng hệ thống hỏi đáp tiếng Việt dựa trên các văn bản pháp luật.  
Dữ liệu được thu thập trực tiếp từ vbpl.vn, sau đó làm sạch, chuẩn hóa và chia nhỏ theo từng Điều, Mục, Chương để có thể truy vấn chính xác hơn.

Mình sử dụng mô hình `AITeamVN/Vietnamese_Embedding` để tạo vector nhúng cho từng đoạn văn bản, lưu trong ChromaDB.  
Khi người dùng đặt câu hỏi, hệ thống sẽ nhúng lại câu truy vấn, tìm các đoạn có độ tương đồng cao, sau đó **rerank thủ công bằng cosine similarity** để đảm bảo độ chính xác trước khi gửi cho mô hình ngôn ngữ `Llama 3.2:3B` (qua Ollama).

Để hệ thống hoạt động ổn định và nhanh hơn, mình có:
- **Giới hạn context theo token** để tránh vượt quá ngữ cảnh mô hình cho phép.  
- **Cơ chế cache query embedding** giúp tốc độ tăng đáng kể khi truy vấn lặp lại.  
- **Xử lý lọc dữ liệu theo Điều luật cụ thể** nếu người dùng nhập dạng “Điều 15”, thay vì tìm mù toàn bộ.  
- **Rerank bằng cosine similarity** giữa embedding truy vấn và embedding của các đoạn văn bản để chọn ra kết quả phù hợp nhất.  
- Kết hợp bước **tokenization tiếng Việt (PyVi)** trước khi embedding để giảm nhiễu và tăng độ khớp ngữ nghĩa.

Toàn bộ pipeline chạy cục bộ, không cần API ngoài.  
