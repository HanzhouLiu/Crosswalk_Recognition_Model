import numpy as np
import math
import matplotlib.pyplot as plt
import math_tools as mt

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

        x1_p = (r_p - y1_p * math.sin(t_p)) / math.cos(t_p)
        x1_n = (r_n - y1_n * math.sin(t_n)) / math.cos(t_n)

        x2_p = (r_p - y2_p * math.sin(t_p)) / math.cos(t_p)
        x2_n = (r_n - y2_n * math.sin(t_n)) / math.cos(t_n)

        x = [x1_p, x2_p, x2_n, x1_n]
        y = [300-y1_p, 300-y2_p, 300-y2_n, 300-y1_n]
        image = plt.fill(x, y)

        return image

    def expand_plt_img(self):
        ymin_pos = self.pos[2, 0]
        ymax_pos = self.pos[3, 0]
        ymin = min(self.pos[2, 0], self.neg[2, 0])
        ymax = max(self.pos[3, 0], self.neg[3, 0])

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

        plt.fill(x, y)

    def expand(self):
        ymin_pos = self.pos[2, 0]
        ymax_pos = self.pos[3, 0]
        ymin = min(self.pos[2, 0], self.neg[2, 0])
        ymax = max(self.pos[3, 0], self.neg[3, 0])

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

        # x = [x1_pos, x2_pos, x2_neg, x1_neg]
        # y = [300 - ymin, 300 - ymax, 300 - ymax, 300 - ymin]

        # image = plt.fill(x, y)
        # pos_endpoints: dtype = np array
        pos_endpoints = np.array([[x1_pos, ymin], [x2_pos, ymax]])
        neg_endpoints = np.array([[x1_neg, ymin], [x2_neg, ymax]])
        # up_edges: dtype = list of tuples
        up_edge = [(x2_pos, ymax), (x2_neg, ymax)]
        down_edge = [(x1_pos, ymin), (x1_neg, ymin)]
        return pos_endpoints, neg_endpoints, up_edge, down_edge

    def vanishing_point(self):
        pos_endpoints = [[self.expand()[0][0, 0], self.expand()[0][0, 1]], 
        [self.expand()[0][1, 0], self.expand()[0][1, 1]]]
        neg_endpoints = [[self.expand()[1][0, 0], self.expand()[1][0, 1]], 
        [self.expand()[1][1, 0], self.expand()[1][1, 1]]]
        x0, y0 = mt.line_intersection(pos_endpoints, neg_endpoints)

        return (x0, y0)