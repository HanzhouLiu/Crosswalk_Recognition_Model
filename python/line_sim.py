import math
import numpy as np
import angle_diff


def line_sim(seg1, seg2, eps_theta, eps_dist):
    """:argument
    seg1, seg2: [[rho], [theta], [ymin], [ymax], [polarity]]
    rst:        1: Yes  0: No
    """
    #rst = 1
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

    rho2 = seg2[0, 0]
    theta2 = seg2[1, 0]
    ymin2 = seg2[2, 0]
    ymax2 = seg2[3, 0]

    # x = (rho[0, 0] - y * math.sin(theta)) / math.cos(theta)
    xmin1 = (rho1 - ymin1 * math.sin(theta1)) / math.cos(theta1)
    xmax1 = (rho1 - ymax1 * math.sin(theta1)) / math.cos(theta1)
    xmid1 = (xmin1 + xmax1) / 2

    xmin2 = (rho2 - ymin2 * math.sin(theta2)) / math.cos(theta2)
    xmax2 = (rho2 - ymax2 * math.sin(theta2)) / math.cos(theta2)
    xmid2 = (xmin2 + xmax2) / 2

    if xmax1 - xmin1 == 0:
        x_intcp1 = xmax1
    else:
        b1 = ymin1 - xmin1 * (ymax1 - ymin1) / (xmax1 - xmin1)
        k1 = (ymax1 - ymin1) / (xmax1 - xmin1)
        x_intcp1 = -b1 / k1

    if xmax2 - xmin2 == 0:
        x_intcp2 = xmax2
    else:
        b2 = ymin2 - xmin2 * (ymax2 - ymin2) / (xmax2 - xmin2)
        k2 = (ymax2 - ymin2) / (xmax2 - xmin2)
        x_intcp2 = -b2 / k2

    diff = angle_diff.angle_diff(seg1[1, 0], seg2[1, 0])
    if diff < eps_theta and abs(x_intcp1 - x_intcp2) < eps_dist:
        rst = 1
    """
    if abs(seg1_xmid - seg2_xmid) > eps_dist:
        rst = 0
    """

    return rst
