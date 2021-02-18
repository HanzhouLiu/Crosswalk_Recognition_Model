import math
import numpy as np
import hd
import matplotlib.path as matpath

def proximity(seg1, seg2, eps_theta, imm, grayscale, ratio):
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

    score = 0

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

    y0_seg1 = seg1[5, 0]
    y0_seg2 = seg2[5, 0]

    y_range1 = min(y1_seg1, y1_seg2)
    y_range2 = max(y2_seg1, y2_seg2)
    y_range = np.arange(y_range1, y_range2 + 0.5, 0.5)

    # rule1:
    # the polarity of two segments should be opposite
    # the two segments should be in the same y0_region
    if p_seg1 + p_seg2 == 0 and y0_seg1 == y0_seg2:
        score = 1
    # rule2:
    # the two segments should be nearly parallel
    if score == 1:
        if abs(t_seg1 - t_seg2) > eps_theta:
            score = 0
    # rule3:
    # seg2 should be located at the right side of seg1
    if score == 1:
        for y0 in y_range:
            x_seg1 = (r_seg1 - y0 * math.sin(t_seg1)) / math.cos(t_seg1)
            x_seg2 = (r_seg2 - y0 * math.sin(t_seg2)) / math.cos(t_seg2)
            x_range = x_seg2 - x_seg1
            if x_range <= 0:
                score = 0
    # rule4:
    # the two segments should be nearly collinear
    if score == 1:
        A = np.array([x1_seg1, y1_seg1])
        B = np.array([x2_seg1, y2_seg1])
        C = np.array([x1_seg2, y1_seg2])
        D = np.array([x2_seg2, y2_seg2])
        hd1, hd2 = hd.hd(A, B, C, D)

        if hd1 < 2 or hd1 > 60:
            score = 0
        if hd2 < 2 or hd2 > 60:
            score = 0
    
    # use contains_points lib to count bight pixels inside a polygon
    if score == 1:
        x1_seg1 = int(x1_seg1)
        x2_seg1 = int(x2_seg1)
        x1_seg2 = int(x1_seg2)
        x2_seg2 = int(x2_seg2)
        y1_seg1 = int(y1_seg1)
        y2_seg1 = int(y2_seg1)
        y1_seg2 = int(y1_seg2)
        y2_seg2 = int(y2_seg2)
        PTS0 = (min(x1_seg1, x2_seg1, x2_seg2, x1_seg2), min(y1_seg1, y1_seg2))
        PTS1 = (min(x1_seg1, x2_seg1, x2_seg2, x1_seg2), max(y2_seg1, y2_seg2))
        PTS2 = (max(x1_seg1, x2_seg1, x2_seg2, x1_seg2), max(y2_seg1, y2_seg2))
        PTS3 = (max(x1_seg1, x2_seg1, x2_seg2, x1_seg2), min(y1_seg1, y1_seg2))
        xmin = PTS0[0]
        ymin = PTS0[1]
        xmax = PTS2[0]
        ymax = PTS2[1]
        pts0 = (x1_seg1, y1_seg1)
        pts1 = (x2_seg1, y2_seg1)
        pts2 = (x1_seg2, y1_seg2)
        pts3 = (x2_seg2, y2_seg2)
        polygon = [pts0, pts1, pts2, pts3]
        list_points = []
        # rows = max(y2_seg1, y2_seg2)-min(y1_seg1, y1_seg2)+1
        rows = PTS1[1]-PTS0[1]+1
        # cols = max(x1_seg1, x2_seg1, x2_seg2, x1_seg2)-min(x1_seg1, x2_seg1, x2_seg2, x1_seg2)+1
        cols = PTS2[0]-PTS1[0]+1
        for j in range(PTS2[0]-PTS1[0]+1):
            for i in range(PTS1[1]-PTS0[1]+1):
                # idx = j*(max(x2_seg2, x1_seg2)-min(x1_seg1, x2_seg1)+1)+i+1-1
                c = PTS0[0]+j
                r = PTS0[1]+i
                # sublist_points = [r, c]
                list_points.append([r, c])
        path = matpath.Path(polygon)
        inside = path.contains_points(list_points)  # inside: [True, False, ...] type: list

        arr_points = np.zeros([rows, cols])

        bright_area = 0
        dark_area = 0
        for k in range(len(inside)):
            # i = k // rows + min(y1_seg1, y1_seg2)
            j = k // rows
            # j = k % rows + min(x1_seg1, x2_seg1)
            i = k % rows
            if not list_points[k]:
                arr_points[i, j] = 0
            else:
                arr_points[i, j] = 1
                rr = i + PTS0[1]
                cc = j + PTS0[0]
                if imm[299-rr, cc] <= grayscale:
                    bright_area = bright_area+1
                else:
                    dark_area = dark_area+1
        if dark_area/(bright_area+dark_area) > ratio:
            score = 0

    return score
