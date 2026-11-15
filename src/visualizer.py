import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def plot_spatial_clusters_map(coords_deg, labels, core_indices=None, save_path=None):
    """
    Ve ban do khong gian phan bo cac cum xung dot toan cau.
    Phan tach ro rang 3 tang: Diem loi (Core), Diem bien (Border) va Diem nhieu (Noise).
    """
    if save_path is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_path = os.path.join(BASE_DIR, "outputs", "figures", "cluster_map.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(15, 10))

    # 1. Lop 1 (Duoi cung): Diem Nhieu (Noise) - mau xam nhat, hat nho lam phong nen
    noise_mask = (labels == -1)
    plt.scatter(coords_deg[noise_mask, 1], coords_deg[noise_mask, 0], 
                c='#B0B0B0', s=6, alpha=0.35, edgecolors='none')

    # Neu co danh sach Diem loi (core_indices), tach rieng Diem loi va Diem bien
    if core_indices is not None:
        core_mask = np.zeros(len(labels), dtype=bool)
        core_mask[core_indices] = True
        border_mask = (labels != -1) & (~core_mask)

        # 2. Lop 2 (O giua): Diem bien (Border) - hat nho, cung mau cum, the hien vanh dai mep
        if np.any(border_mask):
            plt.scatter(coords_deg[border_mask, 1], coords_deg[border_mask, 0], 
                        c=labels[border_mask], cmap='tab20', s=12, alpha=0.75, edgecolors='none')

        # 3. Lop 3 (Tren cung): Diem loi (Core) - hat to, vien mong, the hien trung tam mat do cao
        if np.any(core_mask):
            plt.scatter(coords_deg[core_mask, 1], coords_deg[core_mask, 0], 
                        c=labels[core_mask], cmap='tab20', s=24, alpha=0.9, edgecolors='black', linewidths=0.3)

        # Tao chu thich chi tiet cho tung loai diem va so luong thuc te
        plt.scatter([], [], c='#333333', s=30, edgecolors='black', linewidths=0.5, 
                    label=f'Điểm lõi (Core: {np.sum(core_mask):,} điểm)')
        plt.scatter([], [], c='#666666', s=12, edgecolors='none', 
                    label=f'Điểm biên (Border: {np.sum(border_mask):,} điểm)')
        plt.scatter([], [], c='#B0B0B0', s=8, alpha=0.6, edgecolors='none', 
                    label=f'Nhiễu ngoại lai (Noise: {np.sum(noise_mask):,} điểm)')
    else:
        # Truong hop khong truyen core_indices, ve chung toan bo cum
        cluster_mask = (labels != -1)
        plt.scatter(coords_deg[cluster_mask, 1], coords_deg[cluster_mask, 0], 
                    c=labels[cluster_mask], cmap='tab20', s=18, alpha=0.85, edgecolors='none')
        plt.scatter([], [], c='#2B5B84', s=20, label=f'Điểm thuộc cụm ({np.sum(cluster_mask):,} điểm)')
        plt.scatter([], [], c='#B0B0B0', s=8, alpha=0.6, label=f'Nhiễu ({np.sum(noise_mask):,} điểm)')

    plt.title("BẢN ĐỒ PHÂN CỤM ĐIỂM NÓNG XUNG ĐỘT TOÀN CẦU (DBSCAN FROM SCRATCH)", fontsize=14, fontweight='bold')
    plt.xlabel("Kinh độ (Longitude)", fontsize=11)
    plt.ylabel("Vĩ độ (Latitude)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.legend(loc='lower left', fontsize=10, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Da xuat ban do phan cum tai: {save_path}")

def plot_cluster_statistics(labels, save_path=None):
    """Ve bieu do Top 10 cum lon nhat va ty le giua cum voi nhieu."""
    if save_path is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_path = os.path.join(BASE_DIR, "outputs", "figures", "cluster_analysis.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    counts = Counter(labels)
    n_noise = counts.get(-1, 0)
    n_clustered = len(labels) - n_noise

    top_10 = [item for item in counts.most_common() if item[0] != -1][:10]
    c_ids = [f"Cụm {item[0]}" for item in top_10]
    c_sizes = [item[1] for item in top_10]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Bieu do cot ngang Top 10
    axes[0].barh(c_ids[::-1], c_sizes[::-1], color='#2B5B84', edgecolor='black', linewidth=0.8)
    axes[0].set_title("TOP 10 ĐIỂM NÓNG XUNG ĐỘT QUY MÔ LỚN NHẤT", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Số lượng sự kiện ghi nhận", fontsize=11)
    axes[0].grid(axis='x', linestyle=':', alpha=0.6)

    # 2. Bieu do tron ty le phan bo
    axes[1].pie([n_clustered, n_noise], 
                labels=['Trong cụm tập trung', 'Sự kiện cá biệt (Nhiễu)'], 
                autopct='%1.1f%%', 
                startangle=140, 
                colors=['#2CA02C', '#D62728'],
                explode=(0, 0.08),
                wedgeprops={'edgecolor': 'black', 'linewidth': 0.8})
    axes[1].set_title("TỶ LỆ PHÂN BỐ GIỮA CỤM VÀ NHIỄU", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Da xuat bieu do thong ke tai: {save_path}")

def export_interactive_map_folium(csv_path, labels, output_html=None, max_points=2500):
    """Xuat ban do tuong tac HTML su dung Folium va CartoDB."""
    try:
        import folium
    except ImportError:
        print("Chua cai dat folium. Co the bo qua hoac cai dat bang: pip install folium")
        return

    if output_html is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_html = os.path.join(BASE_DIR, "outputs", "maps", "conflict_hotspots.html")

    data_records = []
    if csv_path.startswith(('http://', 'https://')):
        import urllib.request
        req = urllib.request.Request(csv_path, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            lines = [line.decode('utf-8', errors='ignore') for line in resp.readlines()]
        reader = csv.DictReader(lines)
    else:
        f = open(csv_path, 'r', encoding='utf-8', errors='ignore')
        reader = csv.DictReader(f)

    for i, row in enumerate(reader):
            if i >= len(labels):
                break
            try:
                lat = float(row.get('latitude', 0))
                lon = float(row.get('longitude', 0))
                c_name = row.get('conflict_name', 'Khong ro')
                year = row.get('year', '')
                deaths = row.get('best', '0')
                data_records.append({
                    'lat': lat, 'lon': lon, 'conflict': c_name,
                    'year': year, 'deaths': deaths, 'label': labels[i]
                })
            except (ValueError, TypeError):
                continue

    if len(data_records) > max_points:
        indices = np.random.RandomState(42).choice(len(data_records), max_points, replace=False)
        data_records = [data_records[idx] for idx in indices]

    m = folium.Map(location=[15.0, 30.0], zoom_start=3, tiles='OpenStreetMap')

    for record in data_records:
        lbl = record['label']
        if lbl == -1:
            color = '#808080'
            radius = 2.5
            fill_opacity = 0.35
            cluster_text = "Ngoại lai (Nhiễu)"
        else:
            palette = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33', '#A65628', '#F781BF']
            color = palette[lbl % len(palette)]
            radius = 5.0
            fill_opacity = 0.85
            cluster_text = f"Cụm {lbl}"

        popup_content = (
            f"<b>Sự kiện:</b> {record['conflict']}<br>"
            f"<b>Năm:</b> {record['year']}<br>"
            f"<b>Thương vong ước tính:</b> {record['deaths']}<br>"
            f"<b>Phân loại:</b> {cluster_text}"
        )
        folium.CircleMarker(
            location=[record['lat'], record['lon']],
            radius=radius,
            color=color,
            weight=0.5,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            popup=folium.Popup(popup_content, max_width=250)
        ).add_to(m)

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    m.save(output_html)
    print(f"Da xuat ban do tuong tac tai: {output_html}")

if __name__ == "__main__":
    from dbscan_model import CustomDBSCAN
    from utils import load_and_clean_coordinates, km_to_radians

    csv_file = r"C:\Users\Admin\PycharmProjects\speech-processing-tutorial\DBSCAN\data\processed\GEDEvent_10k_sample.csv"
    coords = load_and_clean_coordinates(csv_file)
    coords_rad = np.radians(coords)

    # Khoi tao mo hinh DBSCAN from scratch
    dbscan = CustomDBSCAN(eps=km_to_radians(50.0), min_samples=10, metrics="haversine")
    labels = dbscan.fit_predict(coords_rad)

    # 1. Xuat ban do phan cum tach biet Core, Border va Noise
    plot_spatial_clusters_map(coords, labels, core_indices=dbscan.core_sample_indices_)

    # 2. Xuat bieu do thong ke Top 10 cum va ty le nhieu
    plot_cluster_statistics(labels)
