import numpy as np
import math
import matplotlib.pyplot as plt

class stripe_operations():

    def __init__(self, pos, neg):
        self.pos = pos
        self.neg = neg

    def shade(self):
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
        r_p = self.pos[0, 0]
        r_n = self.neg[0, 0]
        t_p = self.pos[1, 0]
        t_n = self.neg[1, 0]

        y1_p = self.pos[2, 0]
        y1_n = self.neg[2, 0]
        y2_p = self.pos[3, 0]
        y2_n = self.neg[3, 0]

        # x1 = (r1 - y1 * math.sin(t1)) / math.cos(t1)
        x1_p = (r_p - y1_p * math.sin(t_p)) / math.cos(t_p)
        x1_n = (r_n - y1_n * math.sin(t_n)) / math.cos(t_n)

        x2_p = (r_p - y2_p * math.sin(t_p)) / math.cos(t_p)
        x2_n = (r_n - y2_n * math.sin(t_n)) / math.cos(t_n)

        x = [x1_p, x2_p, x2_n, x1_n]
        y = [300-y1_p, 300-y2_p, 300-y2_n, 300-y1_n]
        image = plt.fill(x, y)

        return image

    def expand(self):
        ymin_pos = self.pos[2, 0]
        ymax_pos = self.pos[3, 0]
        ymin = min(self.pos[2, 0], self.neg[2, 0])
        ymax = max(self.pos[3, 0], self.neg[3, 0])

        # x1_p = (r_p - y1_p * math.sin(t_p)) / math.cos(t_p)
        rho_pos = self.pos[0, 0]
        rho_neg = self.neg[0, 0]
        theta_pos = self.pos[1, 0]
        theta_neg = self.neg[1, 0]
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