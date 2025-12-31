import numpy as np


def sampleFibonacciSpherePoints(
    num_points: int,
    radius: float,
    center: np.ndarray,
) -> np.ndarray:
    """
    使用Fibonacci球面采样生成均匀分布的点

    Args:
        num_points: 点的数量
        radius: 球的半径
        center: 球心位置

    Returns:
        points: shape (num_points, 3) 的点坐标数组
    """
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(num_points):
        # 纵向位置 [-1, 1]
        y = 1 - (i / float(num_points - 1)) * 2
        # 该高度处的半径
        radius_at_y = np.sqrt(1 - y * y)
        # 横向角度
        theta = phi * i

        # 计算球面坐标
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y

        # 缩放到指定半径并平移到中心
        point = np.array([x, y, z]) * radius + center
        points.append(point)

    return np.array(points)
