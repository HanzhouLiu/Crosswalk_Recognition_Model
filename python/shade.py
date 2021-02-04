import numpy as np
import math
import matplotlib.pyplot as plt


def shade(pos_seg, neg_seg):
    """
    (x2_p, y2_p) ***   (x2_n, y2_n)
       /                |
      /                 |
     /                  |
    /                   |
    (x1_p, y1_p) ***   (x1_n, y1_n)
    :param pos_seg:
    :param neg_seg:
    :return: image
    """
    r_p = pos_seg[0, 0]
    r_n = neg_seg[0, 0]
    t_p = pos_seg[1, 0]
    t_n = neg_seg[1, 0]

    y1_p = pos_seg[2, 0]
    y1_n = neg_seg[2, 0]
    y2_p = pos_seg[3, 0]
    y2_n = neg_seg[3, 0]

    # x1 = (r1 - y1 * math.sin(t1)) / math.cos(t1)
    x1_p = (r_p - y1_p * math.sin(t_p)) / math.cos(t_p)
    x1_n = (r_n - y1_n * math.sin(t_n)) / math.cos(t_n)

    x2_p = (r_p - y2_p * math.sin(t_p)) / math.cos(t_p)
    x2_n = (r_n - y2_n * math.sin(t_n)) / math.cos(t_n)

    x = [x1_p, x2_p, x2_n, x1_n]
    y = [300-y1_p, 300-y2_p, 300-y2_n, 300-y1_n]
    image = plt.fill(x, y)

    return image
