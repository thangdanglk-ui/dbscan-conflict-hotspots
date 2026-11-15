import csv
import numpy as np
EARTH_RADIUS_KM = 6371.0088
def km_to_radians(km):
    return km/EARTH_RADIUS_KM
def radians_to_km(radians):
    return radians*EARTH_RADIUS_KM
def load_and_clean_coordinates(csv_path):
    coords = []
    with open(csv_path,"r",encoding="UTF-8",errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader)
        lat_idx = header.index("latitude")
        long_idx = header.index("longitude")
        for row in reader:
            if len(row) > max(lat_idx,long_idx):
                lat_str = row[lat_idx].strip()
                lon_str = row[long_idx].strip()
                if lat_str and lon_str:
                    try:
                        lat = float(lat_str)
                        long = float(lon_str)
                        if -90.0 <= lat <= 90.0 and -180.0 <= long <= 180.0:
                            coords.append([lat,long])
                    except ValueError:
                        continue
    return np.array(coords,dtype = np.float64)
def haversine_distance(p1,p2,in_radians = True):
    if not in_radians:
        lat1,long1 = np.radians(p1[0]),np.radians(p1[1])
        lat2,long2 = np.radians(p2[0]),np.radians(p2[1])
    else:
        lat1,long1 = p1[0],p1[1]
        lat2,long2 = p2[0],p2[1]
    dlat = lat2 - lat1
    dlong = long2 - long1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlong/2)**2
    a = np.clip(a,0.0,1.0)
    c = 2*np.arcsin(np.sqrt(a))
    return c
