# ĐỒ ÁN MÁY HỌC: PHÂN CỤM ĐIỂM NÓNG XUNG ĐỘT VŨ TRANG TOÀN CẦU (UCDP GED)
## THUẬT TOÁN DBSCAN FROM SCRATCH (100% PURE NUMPY)

Dự án nghiên cứu và lập trình từ đầu (from scratch) thuật toán phân cụm dựa trên mật độ **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** thuần túy bằng NumPy, không sử dụng các hàm đóng gói sẵn của thư viện học máy scikit-learn.

Dữ liệu ứng dụng là tập dữ liệu địa lý các sự kiện bạo lực và xung đột vũ trang toàn cầu từ tổ chức **Uppsala Conflict Data Program (UCDP Georeferenced Event Dataset - GED)**.

---

## 1. Cấu trúc thư mục dự án

```text
DBSCAN/
│
├── data/
│   └── processed/
│       ├── GEDEvent_10k_sample.csv       # Tập dữ liệu mẫu 10.000 điểm địa lý chuẩn
│       └── GEDEvent_10k_sample.zip       # Bản nén dự phòng dữ liệu
│
├── src/
│   ├── __init__.py
│   ├── utils.py                          # Tiền xử lý tọa độ và công thức Haversine mặt cầu
│   ├── parameter_tuning.py               # Xác định epsilon tối ưu qua đồ thị k-distance
│   ├── dbscan_model.py                   # Lớp CustomDBSCAN tự lập trình (fit, predict)
│   └── visualizer.py                     # Bản đồ không gian 3 tầng và biểu đồ thống kê
│
├── outputs/
│   ├── figures/                          # Lưu trữ ảnh báo cáo độ phân giải 300 DPI
│   │   ├── k_distance_elbow.png          # Đồ thị khuỷu tay xác định epsilon = 50 km
│   │   ├── cluster_map.png               # Bản đồ phân tầng Điểm lõi, Điểm biên, Nhiễu
│   │   └── cluster_analysis.png          # Top 10 điểm nóng và biểu đồ tròn tỷ lệ nhiễu
│   └── maps/
│       └── conflict_hotspots.html        # Bản đồ tương tác vệ tinh toàn cầu (Folium)
│
├── notebooks/
│   └── dbscan.ipynb                      # Jupyter Notebook thử nghiệm
│
├── main.py                               # Tệp điều phối toàn bộ quy trình từ A đến Z
├── requirements.txt                      # Danh sách các thư viện phụ thuộc
└── README.md                             # Báo cáo tổng quan dự án
```

---

## 2. Các điểm nhấn kỹ thuật nổi bật

1. **Lập trình 100% From Scratch:** Tự cài đặt toàn bộ cấu trúc lớp, ma trận khoảng cách, và thuật toán lan truyền vết mật độ mà không dùng thư viện phân cụm đóng gói sẵn.
2. **Khoảng cách mặt cầu Haversine:** Ứng dụng đúng đắn hình học cầu Trái Đất ($R = 6371.0088\text{ km}$) và góc radian để tính toán khoảng cách địa lý chính xác, triệt tiêu sai số của khoảng cách phẳng Euclid.
3. **Tối ưu hóa thời gian thực thi $O(1)$:** Kết hợp cấu trúc hàng đợi `queue` (list) và tập hợp tra cứu `in_queue` (set) trong quá trình lan truyền BFS, rút ngắn thời gian phân cụm 10.000 điểm từ 40 giây xuống dưới **6.5 giây**!
4. **Sáng tạo phương thức `predict`:** Khắc phục hạn chế lớn của scikit-learn DBSCAN bằng cách phát triển hàm `predict(X_new)` dựa trên khoảng cách tới các Điểm lõi đã lưu trữ.
5. **Cơ sở khoa học chọn tham số:** Sử dụng phương pháp đồ thị $k$-distance để xác định điểm khuỷu tay (elbow point) tại ngưỡng $\varepsilon = 50\text{ km}$ ($0.007848\text{ rad}$) với $\text{min\_samples} = 10$.
6. **Trực quan hóa phân tầng 3 loại điểm:**
   - **Điểm lõi (Core Points):** Hạt lớn màu rực rỡ có viền đen mảnh thể hiện tâm điểm chiến sự.
   - **Điểm biên (Border Points):** Hạt vừa cùng màu cụm thể hiện ranh giới ngoài cùng.
   - **Điểm nhiễu (Noise Points):** Chấm xám nhỏ mờ thể hiện các vụ bạo lực cá biệt, lẻ tẻ.
7. **Bản đồ tương tác Web HTML:** Xuất bản đồ Leaflet.js tương tác cho phép phóng to/thu nhỏ và nhấp xem chi tiết từng sự kiện lịch sử.

---

## 3. Hướng dẫn cài đặt và chạy

### Cài đặt môi trường
Yêu cầu Python 3.10 trở lên. Cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

### Chạy toàn bộ đồ án
Để thực thi toàn bộ chu trình từ đọc dữ liệu, phân cụm đến xuất tất cả biểu đồ và bản đồ:
```bash
python main.py
```

### Chạy từng module độc lập:
* Tìm tham số $\varepsilon$ bằng đồ thị $k$-distance:
  ```bash
  python src/parameter_tuning.py
  ```
* Chạy trực quan hóa độc lập:
  ```bash
  python src/visualizer.py
  ```

---

## 4. Báo cáo kết quả phân cụm thực tế (Tập mẫu 10.000 điểm)

* **Thời gian huấn luyện:** ~6.42 giây.
* **Tổng số cụm điểm nóng tìm thấy:** **98 cụm** tập trung tại các chiến trường lịch sử lớn (Trung Đông, Vùng Ngũ Đại Hồ châu Phi, Nam Á, Đông Âu...).
* **Phân bố các loại điểm:**
  * **Điểm lõi (Core Points):** 7.185 điểm (chiếm 71.85%).
  * **Điểm biên (Border Points):** 556 điểm (chiếm 5.56%).
  * **Điểm nhiễu ngoại lai (Noise):** 2.259 điểm (chiếm 22.59% — lọc sạch các sự kiện lẻ tẻ ở vùng xa xôi).
