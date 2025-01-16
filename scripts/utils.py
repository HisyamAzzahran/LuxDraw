import math

def calculate_distance(point1, point2):
    """Menghitung jarak antara dua titik."""
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
