import csv
import urllib.request
import numpy as np

EARTH_RADIUS_KM = 6371.0088

def km_to_radians(km):
    """Quy đổi khoảng cách kilômét mặt đất sang góc ở tâm (radian)."""
    return km / EARTH_RADIUS_KM

def radians_to_km(radians):
    """Quy đổi góc ở tâm (radian) sang khoảng cách kilômét mặt đất."""
    return radians * EARTH_RADIUS_KM

def load_and_clean_coordinates(source):
    """
    Đọc dữ liệu tọa độ từ tệp nội bộ hoặc trực tiếp từ link URL Internet (GitHub Raw).
    """
    coords = []
    if source.startswith(('http://', 'https://')):
        req = urllib.request.Request(source, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            lines = [line.decode('utf-8', errors='ignore') for line in resp.readlines()]
        reader = csv.reader(lines)
    else:
        with open(source, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        reader = csv.reader(lines)

    header = next(reader)
    lat_idx = header.index("latitude")
    long_idx = header.index("longitude")
    for row in reader:
        if len(row) > max(lat_idx, long_idx):
            lat_str = row[lat_idx].strip()
            lon_str = row[long_idx].strip()
            if lat_str and lon_str:
                try:
                    lat = float(lat_str)
                    long = float(lon_str)
                    if -90.0 <= lat <= 90.0 and -180.0 <= long <= 180.0:
                        coords.append([lat, long])
                except ValueError:
                    continue
    return np.array(coords, dtype=np.float64)

def haversine_distance(p1, p2, in_radians=True):
    """Tính khoảng cách góc (radian) giữa 2 điểm tọa độ trên mặt cầu Trái Đất."""
    if not in_radians:
        lat1, long1 = np.radians(p1[0]), np.radians(p1[1])
        lat2, long2 = np.radians(p2[0]), np.radians(p2[1])
    else:
        lat1, long1 = p1[0], p1[1]
        lat2, long2 = p2[0], p2[1]
    dlat = lat2 - lat1
    dlong = long2 - long1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlong / 2.0)**2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return c
