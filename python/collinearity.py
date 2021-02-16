import math
import numpy as np
import angle_diff
import hd


def collinearity(seg1, seg2, eps_dist):
    """:argument
    seg1, seg2: [[rho], [theta], [ymin], [ymax], [polar], [y0_region]] (add i later)
    rst:        1: Yes  0: No
    """
    # rst = 1
    rst = 0
    """
    if seg1[4, 0] != seg2[4, 0]:
        rst = 0

    diff = angle_diff.angle_diff(seg1[1, 0], seg2[1, 0])
    if diff > eps_theta:
        rst = 0
    """
    rho1 = seg1[0, 0]
    theta1 = seg1[1, 0]
    ymin1 = seg1[2, 0]
    ymax1 = seg1[3, 0]
    polar1 = seg1[4, 0]
    y01 = seg1[5, 0]

    rho2 = seg2[0, 0]
    theta2 = seg2[1, 0]
    ymin2 = seg2[2, 0]
    ymax2 = seg2[3, 0]
    polar2 = seg2[4, 0]
    y02 = seg2[5, 0]

    # x = (rho[0, 0] - y * math.sin(theta)) / math.cos(theta)
    xmin1 = (rho1 - ymin1 * math.sin(theta1)) / math.cos(theta1)
    xmax1 = (rho1 - ymax1 * math.sin(theta1)) / math.cos(theta1)
    # xmid1 = (xmin1 + xmax1) / 2

    xmin2 = (rho2 - ymin2 * math.sin(theta2)) / math.cos(theta2)
    xmax2 = (rho2 - ymax2 * math.sin(theta2)) / math.cos(theta2)
    # xmid2 = (xmin2 + xmax2) / 2

    A = np.array([xmin1, ymin1])
    B = np.array([xmax1, ymax1])
    C = np.array([xmin2, ymin2])
    D = np.array([xmax2, ymax2])

    hd_avr1, hd_avr2 = hd.hd(A, B, C, D)

    if hd_avr1 < eps_dist and hd_avr2 < eps_dist:
        rst = 1

    if y01 != y02 or polar1 != polar2:
        rst = 0

    return rst
