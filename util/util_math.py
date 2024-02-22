"""Help functions for math"""

import math
from typing import Tuple

import numpy as np


def distance_matrix_ln(data: np.array, order: int = 2) -> np.array:
    """Given an order, calculate distance matrix

    Args:
        data (np.array): Input data
        order (int): Non-zero int, or inf. 1 for manhattan
          distance, and 2 for euclidean

    Returns:
        The distance matrix
    """
    if not isinstance(order, int):
        raise ValueError('Input arg [L] needs to be an int')
    data_len = np.shape(data)[0]
    mat = np.zeros(shape=(data_len, data_len))
    for i in range(data_len):
        print(i)
        for j in range(i + 1, data_len):
            distance = np.linalg.norm(data[i] - data[j], ord=order)
            mat[i][j] = mat[j][i] = distance
    return mat


def distance_cosine(v1: np.array, v2: np.array) -> float:
    """Given a pair of input, calculate their (cosine) distance"""
    v1, v2 = v1.flatten(), v2.flatten()
    a = np.dot(v1, v2)
    b = np.linalg.norm(v1)
    _c = np.linalg.norm(v2)
    if b == 0 or _c == 0:
        return 0 if b == _c else 1
    cos_sim = a / (b * _c)
    distance = 1 - cos_sim
    return distance if distance > 0 else 0


def distance_matrix_cos(data: np.array) -> np.array:
    """Calculate (cosine) distance matrix"""
    data_len = np.shape(data)[0]
    mat = np.zeros(shape=(data_len, data_len))

    for i in range(data_len):
        print(i)
        for j in range(i + 1, data_len):
            distance = distance_cosine(data[i], data[j])
            mat[i][j] = mat[j][i] = distance
    return mat


def get_iou(boundary_view: Tuple[int, int, int, int],
            boundary_grid: Tuple[float, float, float, float]) -> float:
    """Calculate IoU (Intersection over Union)

    Args:
        boundary_view: A quadri-tuple (h_left, v_top, h_right, v_bottom),
          indicating the left-top and right-bottom corner of a view
        boundary_grid: (h_left, v_top, h_right, v_bottom) for a hash grid

    Returns:
        IoU value (float)
    """
    v_h1, v_v1, v_h2, v_v2 = boundary_view
    g_h1, g_v1, g_h2, g_v2 = boundary_grid
    garea = (g_h2 - g_h1) * (g_v2 - g_v1)

    h1, v1 = max(v_h1, g_h1), max(v_v1, g_v1)  # left-top
    h2, v2 = min(v_h2, g_h2), min(v_v2, g_v2)  # right-bottom
    w = max(0, (h2 - h1))
    h = max(0, (v2 - v1))

    common_area = w * h
    return common_area / garea


def amp_small_scaler(x: float) -> float:
    """Amplify small signals in UI#

    Args:
        x (float): Input value

    Returns:
        An amplified value (float)
    """
    def ex(_x):
        return math.log(_x + 0.01, 2)
    return (ex(x) - ex(0)) / (ex(1) - ex(0))


def standardization(data: np.array) -> np.array:
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def c(n, m) -> int:
    """Calculate C(n, m) (n > m). This function is
    used in a pair-wise brute search"""
    return int(math.factorial(n) / (math.factorial(m)
                                    * math.factorial(n - m)))


if __name__ == '__main__':
    assert get_iou((1, 1, 8, 8), (0, 0, 10, 10)) == 0.49
    assert get_iou((-2, -2, 4, 5), (0, 0, 10, 10)) == 0.2
    assert get_iou((9, 8, 12, 13), (0, 0, 10, 10)) == 0.02
    assert c(15934, 2) == 126938211
    print('test pass')
