import numpy as np
import math
import matplotlib.pyplot as plt


def expand(pos, neg):
    ymin_pos = pos[2, 0]
    ymax_pos = pos[3, 0]
    ymin = min(pos[2, 0], neg[2, 0])
    ymax = max(pos[3, 0], neg[3, 0])

    # x1_p = (r_p - y1_p * math.sin(t_p)) / math.cos(t_p)
    rho_pos = pos[0, 0]
    rho_neg = neg[0, 0]
    theta_pos = pos[1, 0]
    theta_neg = neg[1, 0]
    x1_pos = (rho_pos - ymin * math.sin(theta_pos)) / math.cos(theta_pos)
    x2_pos = (rho_pos - ymax * math.sin(theta_pos)) / math.cos(theta_pos)
    x1_neg = (rho_neg - ymin * math.sin(theta_neg)) / math.cos(theta_neg)
    x2_neg = (rho_neg - ymax * math.sin(theta_neg)) / math.cos(theta_neg)

    """
    (x2_pos, ymax)---(x2_neg, ymax)
        |               |
        |               |
        |               |
        |               |
        |               |
    (x1_pos, ymin)---(x1_neg, ymin)
    clock wise->
            |  |
             <-
    """

    x = [x1_pos, x2_pos, x2_neg, x1_neg]
    y = [300 - ymin, 300 - ymax, 300 - ymax, 300 - ymin]

    image = plt.fill(x, y)

    return image
