import os
import sys
import numpy as np
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from src.utils import load_and_clean_coordinates, km_to_radians, radians_to_km
except ImportError:
    from utils import load_and_clean_coordinates, km_to_radians, radians_to_km

def compute_k_distance(coords_rad,k=10):
    n = len(coords_rad)
    k_distance = np.zeros(n)
    lat = coords_rad[:,0]
    long = coords_rad[:,1]
    print(f"Tính khoảng cách láng giềng k = {k} cho {n} điểm")
    for i in range(n):
        dlat = lat-lat[i]
        dlong = long-long[i]
        a = np.sin(dlat/2)**2 + np.cos(lat[i])*np.cos(lat)*np.sin(dlong/2)**2
        a = np.clip(a,0.0,1.0)
        dist_i = 2.0*np.arcsin(np.sqrt(a))
        distance_sorted = np.sort(dist_i)
        if len(distance_sorted)>k:
            k_distance[i] = distance_sorted[k]
        else:
            k_distance[i] = distance_sorted[-1]
    return np.sort(k_distance)[::-1]
def plot_k_distance_elbow(k_dist_rad,chosen_eps_km = 50,save_path = None):
    if save_path is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_path = os.path.join(BASE_DIR, "outputs", "figures", "k_distance_elbow.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    k_dist_km = radians_to_km(k_dist_rad)
    chosen_eps_radian = km_to_radians(chosen_eps_km)
    plt.figure(figsize=(10,8))
    plt.plot(range(1,len(k_dist_km)+1),k_dist_km,color="black",linewidth=1.5
             ,label = "Đường cong k-distance")
    plt.axhline(y=chosen_eps_km,color="red",linestyle="--",linewidth=1.2,
                label = f"ngưỡng chọn: eps = {chosen_eps_km} km ({chosen_eps_radian:.5f}) rad")
    plt.title("Biểu đồ k-distance elbow xác định tham số epsilon",fontsize=15,fontweight="bold")
    plt.xlabel("thứ tự dữ liệu (sắp xếp giảm dần",fontsize = 11)
    plt.ylabel("khoảng cách tới láng giềng thứ k (km)",fontsize = 11)
    plt.grid(True,linestyle="--",alpha = 0.5)
    plt.legend(fontsize = 11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Đã xuất đồ thị k-distance tại: {save_path}")

if __name__=="__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_fig_path = os.path.join(BASE_DIR, "outputs", "figures", "k_distance_elbow.png")
    coords = load_and_clean_coordinates(r"C:\Users\Admin\PycharmProjects\speech-processing-tutorial\DBSCAN\data\processed\GEDEvent_10k_sample.csv")
    coords_rad = np.radians(coords)
    k_dist_rad = compute_k_distance(coords_rad,k=5)
    plot_k_distance_elbow(k_dist_rad,save_path = output_fig_path)
