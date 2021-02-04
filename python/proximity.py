import math
import numpy as np
import hd


def proximity(seg1, seg2, eps_theta):
    """:parameter
        seg1/seg2: different line segments, supposed to be pos_segment & neg_segment respectively.
        return score:   higher score means higher likelihood of being a good match.

        seg1                seg2
    (x2_seg1, y2_seg1) ***   (x2_seg2, y2_seg2)
       /                        |
      /                         |   ----------------------------y = y0
     /                          |
    /                           |
    (x1_seg1, y1_seg1) ***   (x1_seg2, y1_seg2)
    """

    score = 1

    r_seg1 = seg1[0, 0]
    r_seg2 = seg2[0, 0]

    t_seg1 = seg1[1, 0]
    t_seg2 = seg2[1, 0]

    y1_seg1 = seg1[2, 0]
    y1_seg2 = seg2[2, 0]
    x1_seg1 = (r_seg1 - y1_seg1 * math.sin(t_seg1)) / math.cos(t_seg1)
    x1_seg2 = (r_seg2 - y1_seg2 * math.sin(t_seg2)) / math.cos(t_seg2)

    y2_seg1 = seg1[3, 0]
    y2_seg2 = seg2[3, 0]
    x2_seg1 = (r_seg1 - y2_seg1 * math.sin(t_seg1)) / math.cos(t_seg1)
    x2_seg2 = (r_seg2 - y2_seg2 * math.sin(t_seg2)) / math.cos(t_seg2)

    p_seg1 = seg1[4, 0]
    p_seg2 = seg2[4, 0]

    y_range1 = min(y1_seg1, y1_seg2)
    y_range2 = max(y2_seg1, y2_seg2)
    y_range = np.arange(y_range1, y_range2 + 0.5, 0.5)

    if p_seg1 == p_seg2 + 2:
        score = 1

    if abs(t_seg1 - t_seg2) > eps_theta:
        score = 0

    for y0 in y_range:
        x_seg1 = (r_seg1 - y0 * math.sin(t_seg1)) / math.cos(t_seg1)
        x_seg2 = (r_seg2 - y0 * math.sin(t_seg2)) / math.cos(t_seg2)
        x_range = x_seg2 - x_seg1
        if x_range <= 0:
            score = 0
    """:cvar
    h1_width = abs(np.dot(np.array([[x1_seg1 - x1_seg2], [y1_seg1 - y1_seg2]]).T,
                          np.array([[math.cos(t_seg1)], [math.sin(t_seg1)]])))
    if h1_width < 5 or h1_width > 50:
        score = 0

    h2_width = abs(np.dot(np.array([[x2_seg1 - x2_seg2], [y2_seg1 - y2_seg2]]).T,
                          np.array([[math.cos(t_seg1)], [math.sin(t_seg1)]])))
    if h2_width < 5 or h2_width > 50:
        score = 0    
    """
    A = np.array([x1_seg1, y1_seg1])
    B = np.array([x2_seg1, y2_seg1])
    C = np.array([x1_seg2, y1_seg2])
    D = np.array([x2_seg2, y2_seg2])
    hd1, hd2 = hd.hd(A, B, C, D)

    if hd1 < 5 or hd1 > 40:
        score = 0
    if hd2 < 5 or hd2 > 40:
        score = 0

    # dover = max(0, min(y3, y4) - max(y1, y2))
    # dnorm = min(y3 - y1, y4 - x2)
    # score = score * dover/dnorm
    # score = score * math.exp(-abs(t1-t2)/c)

    return score
