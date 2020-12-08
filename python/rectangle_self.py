import numpy as np
import math


def rectangle_self(region, G, im):
    """
    :param region:  2 * i 2-D array, 1st line---row#, 2nd line---column#
    :param G:
    :param im:
                    2*N each column as a pixel in the region
    :return:        rect_center 2*1 coordinate
                    rect_angle 2*1 eigenvector of the structure tensor
                    W: Width of the rectangle
                    L: Length of the rectangle
    """
    M, N = G.shape
    cx = 0
    cy = 0
    norm_term = 0
    for i in range(len(region[0, :])):
        r = region[0, i]
        c = region[1, i]
        x = c
        y = M - r - 1
        cx = cx + G[r, c]*x
        cy = cy + G[r, c]*y
        norm_term = norm_term + G[r, c]
    cx = cx / norm_term
    cy = cy / norm_term
    rect_center = np.array([[cx], [cy]])
    mxx = 0
    mxy = 0
    myy = 0
    for i in range(len(region[0, :])):
        r = region[0, i]
        c = region[1, i]
        x = c
        y = M - r - 1
        mxx = mxx + G[r, c] * ((x - cx)**2)
        mxy = mxy + G[r, c] * (x - cx) * (y - cy)
        myy = myy + G[r, c] * ((y - cy)**2)
    Matrix = np.array([[mxx, mxy], [mxy, myy]])
    Matrix = Matrix / norm_term
    # calculate eigenvectors
    w, V = np.linalg.eig(Matrix)
    if w[1] > w[0]:
        v = V[:, 0]
    else:
        v = V[:, 1]
    rect_angle = math.atan2(v[1], v[0])

    # normal vector
    n = np.array([[math.cos(rect_angle)], [math.sin(rect_angle)]])
    n_perp = np.dot(np.array([[0, 1], [-1, 0]]), n)
    pos_W_max = 0
    neg_W_max = 0
    pos_L_max = 0
    neg_L_max = 0
    for i in range(len(region[0, :])):
        r = region[0, i]
        c = region[1, i]
        x = c
        y = M - r - 1
        dw = np.dot((np.array([[x], [y]]) - rect_center).T, n)
        dl = np.dot((np.array([[x], [y]]) - rect_center).T, n_perp)
        if dw > pos_W_max:
            pos_W_max = dw
        elif dw < neg_W_max:
            neg_W_max = dw
        if dl > pos_L_max:
            pos_L_max = dl
        elif dl < neg_L_max:
            neg_L_max = dl

    W = np.maximum(pos_W_max, -neg_W_max) * 2
    L = np.maximum(pos_L_max, -neg_L_max) * 2

    return rect_center, rect_angle, W, L
