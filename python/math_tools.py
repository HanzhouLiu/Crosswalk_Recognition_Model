import numpy as np
import math


def conv2(image, kernel):
    # Iterate through image
    output = np.zeros([image.shape[0] - 1, image.shape[1] - 1])
    empty_arr = np.zeros([2, 2])
    for row in range(image.shape[0] - 1):
        # Exit Convolution
        for col in range(image.shape[1] - 1):
            empty_arr = image[row:row + 2:1, col:col + 2:1]
            inner_product = empty_arr.ravel().dot(kernel.ravel())
            output[row, col] = inner_product
    return output


def atan2(a, b):
    output = np.zeros([a.shape[0], a.shape[1]])
    for row in range(a.shape[0]):
        for col in range(a.shape[1]):
            output[row, col] = math.atan2(a[row, col], b[row, col])
    return output


def sqrt(a, b):
    output = np.zeros([a.shape[0], a.shape[1]])
    for row in range(a.shape[0]):
        for col in range(a.shape[1]):
            output[row, col] = math.sqrt(a[row, col] ** 2 + b[row, col] ** 2)
    return output


def max_element(arr):
    lis = []
    for i in range(arr.shape[0]):
        lis.append(max(arr[i]))
    return max(lis)


def tri_area(A, B, C):
    """
    A, B, C should be vectors (2, )
    """
    x1, y1, x2, y2, x3, y3 = A[0], A[1], B[0], B[1], C[0], C[1]
    S = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    # S = 0.5 * abs(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))
    return S

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def dist(p1, p2):
    diff_x = p1[0] - p2[0]
    diff_y = p1[1] - p2[1]
    d = math.sqrt(diff_x ** 2 + diff_y ** 2)
    return d