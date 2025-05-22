import math

def manhattan_distance(a: tuple, b: tuple) -> float:
    """Tính khoảng cách Manhattan giữa hai điểm"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean_distance(a: tuple, b: tuple) -> float:
    """Tính khoảng cách Euclidean giữa hai điểm"""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def diagonal_distance(a: tuple, b: tuple) -> float:
    """
    Tính khoảng cách Diagonal (Chebyshev) giữa hai điểm
    Phù hợp cho việc di chuyển 8 hướng
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy)
