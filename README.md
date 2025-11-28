# 🚀 ResNet-FAISS-Dedup

**Hệ thống lọc, làm sạch và khử trùng lặp dữ liệu ảnh hiệu năng cao sử dụng Deep Learning (ResNet50) và Tìm kiếm Vector (FAISS).**

> **Lưu ý:** Dự án này được tối ưu hóa đặc biệt cho **Apple Silicon (M-Series)** sử dụng PyTorch MPS (Metal Performance Shaders).

## 📖 Giới thiệu

**ResNet-FAISS-Dedup** là một Data Pipeline mạnh mẽ được thiết kế để xử lý các bộ dữ liệu ảnh thô (Raw Dataset) có quy mô lớn. Ứng dụng giải quyết 3 bài toán cốt lõi của việc chuẩn bị dữ liệu cho AI Training:

1.  **Lọc rác:** Loại bỏ ảnh mờ, quá tối, quá sáng hoặc lỗi file.
2.  **Khử trùng lặp tuyệt đối:** Loại bỏ các file giống hệt nhau (SHA-256, pHash, dHash).
3.  **Khử trùng lặp ngữ nghĩa (Semantic Deduplication):** Sử dụng AI để phát hiện các ảnh chụp cùng một góc độ, nội dung giống nhau \> xx% nhưng khác tên hoặc kích thước.

-----

## 🏗 Kiến trúc hệ thống

Pipeline hoạt động tuần tự qua 5 bước chính:

1.  **🔍 Bước 1: Quality Filter (OpenCV)**
      * Kiểm tra độ nét (Laplacian Variance).
      * Kiểm tra độ sáng trung bình (Mean Brightness).
      * Loại bỏ ảnh lỗi định dạng.
2.  **⚡ Bước 2: Hashing Deduplication**
      * Tính toán 3 lớp mã băm: `SHA-256` (Tuyệt đối), `pHash` (Cấu trúc), `dHash` (Gradient).
      * Sử dụng chiến thuật Map-Reduce để quét trùng lặp tốc độ cao.
3.  **🧠 Bước 3: Deep Learning Embedding (FastReID)**
      * **Model:** ResNet50 (Pre-trained trên Vehicle/ImageNet).
      * Trích xuất đặc trưng ảnh thành vector 2048 chiều.
      * Sử dụng `MPS` (GPU) trên Mac để tăng tốc.
4.  **📐 Bước 4: Normalization**
      * Chuẩn hóa vector L2 (Euclidean Norm) sử dụng NumPy.
5.  **🕸 Bước 5: Clustering & Graph Filtering**
      * **FAISS:** Tìm kiếm các vector tương đồng (Cosine Similarity).
      * **NetworkX:** Xây dựng đồ thị liên thông các ảnh trùng lặp.
      * **Decision Logic:** Trong một nhóm trùng, giữ lại ảnh có độ chi tiết (Detail Score) cao nhất và sắc nét nhất.

-----

## 📊 Hiệu năng thực tế (Benchmarks)

Hệ thống được kiểm thử trên phần cứng:

  * **Machine:** Mac Mini M4 (2024)
  * **Specs:** 10-core CPU, 10-core GPU, 24GB RAM.
  * **Storage:** Ổ cứng rời SSD (External NVMe) qua cổng Thunderbolt 4 (Băng thông tối đa 10Gbps).

### Kịch bản 1: Dataset tiêu chuẩn (Vehicle ReID)

  * **Số lượng:** 116,298 ảnh.
  * **Đặc điểm:** Ảnh crop kích thước cố định \~640x640. Dung lượng tổng \~11.3GB.
  * **Tổng thời gian:** 25 phút 38 giây.
  * **Tốc độ trung bình:** \~75 ảnh/giây.
  * **Kết quả:** Lọc bỏ \~26,000 ảnh rác và trùng lặp.

### Kịch bản 2: Dataset chất lượng cao (High-Res Raw)

  * **Số lượng:** 1,268 ảnh.
  * **Đặc điểm:** Ảnh gốc chưa qua xử lý, kích thước rất lớn (\~13MB/ảnh). Tổng \~17GB.
  * **Tổng thời gian:** \~10 phút.
  * **Tốc độ trung bình:** \~2 ảnh/giây.
  * **Ghi chú:** Tốc độ giảm do chi phí I/O và CPU khi giải nén/resize ảnh độ phân giải cao (4K/8K).

-----

## 🛠 Yêu cầu hệ thống

  * **Hệ điều hành:** macOS (Khuyên dùng Sequoia 15+).
      * *Lưu ý: Code chưa được test thực tế trên Windows/Linux.*
  * **Python:** **3.9** (Bắt buộc để tương thích với `fastreid` và các thư viện cũ).
  * **Phần cứng:** Khuyên dùng Apple Silicon (M1/M2/M3/M4) để tận dụng tăng tốc phần cứng.

-----

## 📥 Cài đặt

### 1\. Clone repository

```bash
git clone https://github.com/ngTamAn111103/ResNet-FAISS-Dedup.git
cd ResNet-FAISS-Dedup
```

### 2\. Thiết lập môi trường (Bắt buộc Python 3.9)

Khuyến khích sử dụng `venv` hoặc `conda`.

```bash
# Kiểm tra phiên bản python
python3 --version 
# Nếu chưa có python 3.9, hãy cài đặt qua Homebrew: brew install python@3.9

# Tạo môi trường ảo
python3.9 -m venv .venv

# Kích hoạt môi trường
source .venv/bin/activate
```

### 3\. Cài đặt thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

### 4\. Tải Model Weights & Config

Vì file weights khá nặng, vui lòng tải thủ công và đặt vào thư mục `configs/`:

  * **Vehicle Weights (.pth):** https://drive.google.com/file/d/1LJ8OWIaYPZjb4KFOwsr4MtcLdM4ApiMF/view?usp=sharing
  * **Config File (.yaml):** https://drive.google.com/file/d/1LJ8OWIaYPZjb4KFOwsr4MtcLdM4ApiMF/view?usp=sharing

Cấu trúc thư mục sau khi tải:

```text
ResNet-FAISS-Dedup/
├── configs/
│   ├── vehicle_weights.pth
│   └── vehicle_config.yaml
├── final.py
└── ...
```

-----

## 🚀 Hướng dẫn sử dụng

### 1\. Cấu hình

Mở file `final.py` và chỉnh sửa các biến đường dẫn:

```python
# ___Đường dẫn___
INPUT_FOLDER = '/path/to/your/raw_dataset'  # Đường dẫn tuyệt đối tới folder ảnh
OUTPUT_BASE = '/path/to/output_result'      # Nơi lưu kết quả

# ___Cấu hình lọc (Tùy chỉnh theo dataset)___
BLUR_THRESHOLD = 50.0       # Ngưỡng mờ (Cao = Khắt khe)
THRESHOLD_FAISS = 0.7       # Ngưỡng giống nhau AI (0.7 - 0.9)
```

### 2\. Chạy ứng dụng

```bash
python final.py
```

### 3\. Kết quả đầu ra

Sau khi chạy xong, thư mục `OUTPUT_BASE` sẽ có cấu trúc:

  * `blur/`, `dark/`, `bright/`: Chứa các ảnh kém chất lượng bị loại.
  * `duplicates/`: Chứa ảnh trùng lặp tuyệt đối (Hash).
  * `similar/`: Chứa ảnh trùng lặp ngữ nghĩa (AI detection).
  * `cleaning_report.html`: **Báo cáo trực quan (Xem chi tiết bên dưới).**
  * `cleaning_log.json`: Log dạng text.

-----

## 📈 Báo cáo trực quan (HTML Report)

Ứng dụng tự động sinh ra file `cleaning_report.html`. Bạn có thể mở bằng trình duyệt để xem lại:

  * Thống kê số lượng ảnh bị loại.
  * So sánh song song (Side-by-side) cặp ảnh: Ảnh được giữ lại (Kept) và Ảnh bị xóa (Deleted).
  * Hiển thị lý do xóa và điểm số chênh lệch.

*(Hãy thay thế hình ảnh này bằng screenshot thực tế file report của bạn)*

-----

## 🧩 Dataset tham khảo

Dự án sử dụng bộ dữ liệu mẫu (hoặc tương tự) từ Kaggle:

  * **Link Dataset:** https://www.kaggle.com/datasets/anonynov03/vietnamese-license-plate-2025-v1-1/data

-----

## ⚠️ Lưu ý quan trọng

1.  **Ảnh chất lượng cao (High-Res):** Khi chạy với ảnh Raw/4K (trên 10MB/ảnh), tốc độ sẽ chậm đi đáng kể ở bước 5 (AI Filtering) do thuật toán tính toán độ chi tiết (Canny Edge) đang chạy trên độ phân giải gốc. Phiên bản tối ưu cho High-Res sẽ được cập nhật sau.
2.  **Backup dữ liệu:** Mặc dù code được thiết kế an toàn (chỉ di chuyển file, không xóa vĩnh viễn), hãy luôn backup dữ liệu gốc trước khi chạy.

-----

## 🤝 Đóng góp

Mọi ý kiến đóng góp hoặc báo lỗi vui lòng tạo Issue hoặc Pull Request.

**Author:** Nguyễn Tâm An 

```
```
