import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urlparse, parse_qs
import concurrent.futures
from tqdm import tqdm
import threading

# --- PHẦN 1: CẤU HÌNH ---
INPUT_FILE = "../data/vbpl_tat_ca_links.json"
OUTPUT_FILE = "../data/vbpl_data_hoan_chinh.jsonl"
ERROR_LOG_FILE = "../data/error_log.txt"  # File để ghi lại các link lỗi
MAX_WORKERS = 8

# --- PHẦN 2: HÀM XỬ LÝ LÕI ---
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Referer': 'https://vbpl.vn/'
})

# Sử dụng một Lock để đảm bảo việc ghi file log lỗi từ nhiều luồng là an toàn
error_lock = threading.Lock()

def log_error(url, error_message):
    """Hàm an toàn để ghi lỗi từ nhiều luồng."""
    with error_lock:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"URL: {url}\nError: {error_message}\n---\n")

def scrape_document_details(doc_info):
    content_url = doc_info['url']
    doc_attributes = {}
    doc_content = ""

    try:
        # Trích xuất ItemID
        parsed_url = urlparse(content_url)
        item_id = parse_qs(parsed_url.query).get('ItemID', [None])[0]
        if not item_id:
            log_error(content_url, "Không tìm thấy ItemID.")
            return None

        attribute_url = f"https://vbpl.vn/tw/Pages/vbpq-thuoctinh.aspx?dvid=13&ItemID={item_id}"

        # --- Lấy thuộc tính ---
        response_attr = None
        for attempt in range(3):
            try:
                response_attr = session.get(attribute_url, timeout=15)
                if 400 <= response_attr.status_code < 500:
                    log_error(attribute_url, f"Client Error {response_attr.status_code}. Không thử lại.")
                    return None
                if response_attr.status_code == 200:
                    break
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    log_error(attribute_url, f"Lỗi mạng sau 3 lần thử: {e}")
                    return None
                time.sleep(2)
        
        if not response_attr:
            return None

        response_attr.encoding = 'utf-8'
        soup_attr = BeautifulSoup(response_attr.text, 'html.parser')
        
        important_attributes = [
            "Số ký hiệu", "Loại văn bản", "Ngày ban hành", "Ngày có hiệu lực", "Tình trạng hiệu lực",
            "Cơ quan ban hành/ Chức danh / Người ký", "Nguồn thu thập", "Ngày đăng công báo", "Phạm vi"
        ]
        attribute_table = soup_attr.select_one("div.vbProperties table")
        if attribute_table:
            for row in attribute_table.find_all('tr'):
                columns = row.find_all('td')
                if len(columns) >= 2:
                    label1 = columns[0].text.strip().replace(':', '')
                    if label1 in important_attributes:
                        doc_attributes[label1] = columns[1].text.strip()
                    if len(columns) >= 4:
                        label2 = columns[2].text.strip().replace(':', '')
                        if label2 in important_attributes:
                            doc_attributes[label2] = columns[3].text.strip()
            status_element = soup_attr.select_one(".vbStatus")
            if status_element:
                doc_attributes["Tình trạng hiệu lực"] = status_element.text.strip().replace('Tình trạng hiệu lực:', '').strip()
        
        # --- Lấy nội dung ---
        response_content = None
        for attempt in range(3):
            try:
                response_content = session.get(content_url, timeout=15)
                if 400 <= response_content.status_code < 500:
                    log_error(content_url, f"Client Error {response_content.status_code}. Không thử lại.")
                    return None
                if response_content.status_code == 200:
                    break
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    log_error(content_url, f"Lỗi mạng sau 3 lần thử: {e}")
                    return None
                time.sleep(2)

        if not response_content:
            return None

        response_content.encoding = 'utf-8'
        soup_content = BeautifulSoup(response_content.text, 'html.parser')
        content_area = soup_content.find('div', id='toanvancontent')
        if content_area:
            doc_content = content_area.get_text(separator='\n', strip=True)
        else:
            log_error(content_url, "Không tìm thấy div#toanvancontent.")

        return {
            'tieu_de': doc_info['title'],
            'url_goc': content_url,
            'thuoc_tinh': doc_attributes,
            'noi_dung': doc_content
        }

    except Exception as e:
        log_error(content_url, f"Lỗi không lường trước: {e}")
        return None

# --- PHẦN 3: LOGIC CHÍNH ---

if __name__ == "__main__":
    print("--- BẮT ĐẦU GIAI ĐOẠN 2 (BỀN BỈ) ---")
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            all_links_to_process = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{INPUT_FILE}'.")
        exit()

    scraped_urls = set()
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                scraped_urls.add(json.loads(line)['url_goc'])
    except FileNotFoundError:
        print("Không tìm thấy file output, sẽ bắt đầu tạo file mới.")

    remaining_links = [link for link in all_links_to_process if link['url'] not in scraped_urls]
    print(f"Tổng số link cần xử lý: {len(all_links_to_process)}")
    print(f"Số link đã xử lý: {len(scraped_urls)}")
    print(f"Số link còn lại cần cào: {len(remaining_links)}")

    if not remaining_links:
        print("Tất cả các link đã được xử lý. Hoàn thành!")
        exit()
    
    error_count = 0

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            progress_bar = tqdm(executor.map(scrape_document_details, remaining_links), total=len(remaining_links), desc="Đang cào dữ liệu")
            
            for result in progress_bar:
                if result:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                else:
                    error_count += 1
                progress_bar.set_postfix(errors=error_count)

    print(f"\n--- HOÀN THÀNH! ---")
    print(f"Đã xử lý xong các link còn lại.")
    print(f"Số lượng lỗi gặp phải: {error_count}. Chi tiết xem tại '{ERROR_LOG_FILE}'")
    print(f"Dữ liệu được lưu trong '{OUTPUT_FILE}'")
