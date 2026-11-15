import os
import sys
import time
import numpy as np
from collections import Counter

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


from src.utils import load_and_clean_coordinates, km_to_radians
from src.dbscan_model import CustomDBSCAN
from src.visualizer import plot_spatial_clusters_map, plot_cluster_statistics, export_interactive_map_folium

def run_project_pipeline():
    print("=" * 65)
    print("  ĐỒ ÁN PHÂN CỤM ĐIỂM NÓNG XUNG ĐỘT TOÀN CẦU (DBSCAN FROM SCRATCH)  ")
    print("=" * 65)

    # 1. Chuẩn bị thư mục đầu ra
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/maps", exist_ok=True)

    # 2. Nạp dữ liệu sạch
    data_file = r"data/processed/GEDEvent_10k_sample.csv"
    coords_deg = load_and_clean_coordinates(data_file)
    print(f"\n[Bước 1] Nạp dữ liệu thành công: {len(coords_deg):,} điểm tọa độ hợp lệ.")

    # Chuyển đổi tọa độ sang radian để phục vụ khoảng cách mặt cầu Haversine
    coords_rad = np.radians(coords_deg)

    # 3. Thiết lập siêu tham số từ đồ thị k-distance
    eps_km = 50.0
    eps_rad = km_to_radians(eps_km)
    min_pts = 10

    print(f"\n[Bước 2] Khởi tạo CustomDBSCAN:")
    print(f"         - Bán kính quét epsilon: {eps_km} km ({eps_rad:.6f} rad)")
    print(f"         - Số láng giềng tối thiểu min_samples: {min_pts}")
    print(f"         - Thước đo khoảng cách: Haversine mặt cầu")

    model = CustomDBSCAN(eps=eps_rad, min_samples=min_pts, metrics='haversine')

    # 4. Huấn luyện và phân cụm
    print(f"\n[Bước 3] Đang tiến hành phân cụm trên {len(coords_rad):,} điểm dữ liệu...")
    start_time = time.time()
    labels = model.fit_predict(coords_rad)
    elapsed = time.time() - start_time
    print(f"         -> Huấn luyện hoàn tất trong {elapsed:.2f} giây!")

    # 5. Thống kê chuyên sâu kết quả phân cụm
    counts = Counter(labels)
    n_noise = counts.get(-1, 0)
    n_clusters = len(counts) - (1 if -1 in counts else 0)
    core_count = len(model.core_sample_indices_)
    border_count = len(labels) - n_noise - core_count
    noise_ratio = (n_noise / len(labels)) * 100

    print("\n" + "-" * 40)
    print("         BÁO CÁO THỐNG KÊ KẾT QUẢ:")
    print("-" * 40)
    print(f"  * Tổng số cụm điểm nóng tìm thấy : {n_clusters} cụm")
    print(f"  * Số điểm lõi (Core Points)       : {core_count:,} điểm")
    print(f"  * Số điểm biên (Border Points)    : {border_count:,} điểm")
    print(f"  * Số sự kiện cá biệt (Noise)      : {n_noise:,} điểm ({noise_ratio:.2f}%)")
    print("-" * 40)

    # 6. Xuất các sản phẩm trực quan hóa
    print(f"\n[Bước 4] Xuất các biểu đồ trực quan hóa...")
    plot_spatial_clusters_map(coords_deg, labels, core_indices=model.core_sample_indices_, 
                              save_path="outputs/figures/cluster_map.png")
    plot_cluster_statistics(labels, save_path="outputs/figures/cluster_analysis.png")
    export_interactive_map_folium(data_file, labels, output_html="outputs/maps/conflict_hotspots.html")

    # 7. Thử nghiệm dự báo điểm mới (predict)
    print(f"\n[Bước 5] Thử nghiệm phương thức predict cho 2 sự kiện mới:")
    test_points_deg = np.array([
        [33.5138, 36.2765],  # Damascus, Syria (vùng chiến sự dày đặc)
        [0.0, -140.0]        # Giữa Thái Bình Dương (vùng không có người)
    ])
    test_points_rad = np.radians(test_points_deg)
    preds = model.predict(test_points_rad)
    print(f"  * Điểm Damascus (33.51, 36.28)  -> Dự báo vào Cụm: {preds[0]}")
    print(f"  * Điểm Thái Bình Dương (0, -140) -> Dự báo vào Cụm: {preds[1]} (Nhiễu/An toàn)")

    print("\n" + "=" * 65)
    print("       PIPELINE ĐÃ HOÀN TẤT THÀNH CÔNG RỰC RỠ!       ")
    print("=" * 65)

if __name__ == '__main__':
    run_project_pipeline()