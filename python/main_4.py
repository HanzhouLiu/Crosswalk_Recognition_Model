"""
    This version of codes introduces various zones to as
    a filter.
"""

import cv2 as cv
import numpy as np
import math_tools as mt
import math
import region_grow
import rectangle_self
import polarity
import proximity
import collinearity
import shade
import expand
import matplotlib.pyplot as plt


def main():
    # Hyper-parameters
    M = 300
    N = 400

    print("zebra-crossing detection algorithm")
    filename = '../Samples/image003.jpg'
    img = cv.imread(filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # white pixel == 255
    # black pixel == 0
    img = cv.resize(gray, (N, M))
    # np.savetxt('gray_img.txt', img, fmt='%d')
    imm = img
    img = 255 - img
    # cv.imshow("img", imm)
    # cv.waitKey(0)
    # cv.imshow("img", img)
    # cv.waitKey(0)

    """
    1st step: GaussianBlur
    Filter the image with a Gaussian filter with standard deviation of 0.6/0.8
    Therefore, a Gaussian kernel is used.
    It's done with the function cv2.GaussianBlur()
    """
    img_blur = cv.GaussianBlur(img, sigmaX=0.6 / 0.8, sigmaY=0.6 / 0.8, ksize=(3, 3))
    cv.addWeighted(img_blur, 1.5, img, -0.5, 0, img)

    # Visualization1: Preprocessed image
    cv.imshow("img", imm)
    cv.waitKey(0)

    """
    2nd step: Compute Gradient
    """
    [w, h] = img.shape
    # w=image width, h=image height
    zero_padded_img = np.zeros((w + 1, h + 1))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            zero_padded_img[x, y] = img[x, y]
    # zero_padded_img[:w, :h] = img

    x_kernel = np.array([[1, -1], [1, -1]])
    y_kernel = np.array([[1, 1], [-1, -1]])

    gx = mt.conv2(zero_padded_img, x_kernel) / 2
    gy = mt.conv2(zero_padded_img, y_kernel) / 2
    # for i in range(gx.shape[0]):
    #     for j in range(gx.shape[1]):
    #         if gx[i, j] < 0:
    #             print(gx[i, j])
    """
    the second method to calculate 2d convolution
    by using cv2.sobel
    *******************************************
    gx_64f = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=-1)
    abs_gx_64f = np.absolute(gx_64f)
    gx = np.uint8(abs_gx_64f)
    cv.imshow("imgsobel", gx)
    cv.waitKey(0)

    gy_64f = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=-1)
    abs_gy_64f = np.absolute(gy_64f)
    gy = np.uint8(abs_gy_64f)   
    """

    # plt.imshow(zero_padded_img, cmap='gray')
    # plt.show()

    # print(gx)
    # Saving the array in a text file
    # np.savetxt("file1.txt", gx)

    # Displaying the contents of the text file
    # content = np.loadtxt('file1.txt')
    # print("\nContent in file1.txt:\n", content)

    angle = mt.atan2(-gy, gx)
    # atan2 allows us to calculate the arctangent of all four quadrants
    G = mt.sqrt(gx, gy)

    """
    3rd step: Sort Gradient 
    """
    tau = math.pi / 8
    q = 2
    threshold = q / math.sin(tau)
    max_gradient = mt.max_element(G)

    # 1024 bins [2, n, 1024]
    bins = np.zeros((1024, 2, 10240), dtype=int)
    bin_ends = np.ones((1024, 1), dtype=int)
    bin_centers = np.arange(max_gradient, 0, -max_gradient / 1024)

    for c in range(1, N - 1):
        for r in range(1, M - 1):
            if G[r, c] > threshold:
                dist = abs(bin_centers - G[r, c])
                # print(dist)
                # idx = dist.index(min(dist))
                # using np.where
                idx = np.where(dist == dist.min())
                # print(idx)
                bins[idx, :, bin_ends[idx]] = np.array([r, c]).T
                bin_ends[idx] = bin_ends[idx] + 1
    # print(type(bin_ends[1, 0]))

    """
    experiment1: generate gradient plot
    """
    """
    for r in range(150, M-20):
        c = np.linspace(0, N-1, N, dtype=int)
        # print(c)
        fig = plt.figure()
        plt.plot(c, G[r, c])
        path = '../data/gradient_plot/' + str(r) + '.png'
        plt.savefig(path)
    """

    # kmenas algorithm
    # kmeans = KMeans(n_clusters=2).fit(G)
    # pred = kmeans.predict(data)
    # print(pred)
    # print edge points
    """
    for r in range(M):
        edge = []
        for c in range(N-1):
            if G[r, c] + 20 < G[r, c+1]:
                edge.append(c)
        print(edge)
    """
    """
    4th step: Region Grow
    """
    # output for next step
    output_regions = {}
    output_rectangles = {}

    count = 0
    # parameter
    tau = math.pi / 8

    # constants
    used = 1
    not_used = 0

    status = np.ones((M, N), dtype=int)
    for c in range(1, N):
        # print(c)
        for r in range(1, M):
            # condition to judge a point in x or y direction line segment
            if G[r, c] > threshold and abs(gx[r, c]) > 0:
                status[r, c] = 0  # everything is good here
                # print(1)

    # which bin is active, starting from 0
    # itr = 0

    # visualization 1: show original img as background

    for itr in range(1024):
        for i in range(int(bin_ends[itr, 0])):  # everything is good here
            # print(1)
            pixel = bins[itr, :, i]
            # print(pixel.shape)
            r = pixel[0]
            c = pixel[1]
            if status[r, c] == not_used:
                region, status = region_grow.region_grow(angle, pixel, tau, status)  # everything is good here
                # print(len(region[0, :]))
                if len(region[0, :]) < 20:
                    continue
                rect_center, rect_angle, W, L = rectangle_self.rectangle_self(region, G, img)
                # print(rect_center.shape) 2 * 1(r * c)
                # print(len(region[0, :]))
                """
                which pattern of lines are selected can be defined here:
                rect_angle(pseudo rect_angle) + pi/2 = the true rect_angle
                remove all line segments whose slope less than pi/4
                """
                if abs(abs(rect_angle) - math.pi / 2) < math.pi / 8:
                    continue
                """
                print all pseudo rect_angle
                """
                # print(rect_angle)
                output_regions[count] = region
                output_rectangles[count] = np.concatenate((rect_center, np.array([[rect_angle], [W], [L]])), axis=0)
                # print(1)
                count = count + 1
                # print('count=', count)
        # print(bin_ends[itr][0])
        # print(type(bin_ends[itr]))
    """
    5th step: Group line segments
    this step is to create line segments
    [rho, theta, xmin, xmax, polarity]
    theta in region [0, pi]
    """
    segment_functions = {}
    for i in range(count):
        p0 = np.array([output_rectangles[i][0], output_rectangles[i][1]])
        n_angle = output_rectangles[i][2]
        # print(angle.shape)
        L = output_rectangles[i][4]
        n = np.array([[math.cos(n_angle)], [math.sin(n_angle)]])
        # print(n.shape)
        # print(p0.shape)
        rho = np.dot((p0 - np.array([[0], [0]])).T, n)
        # print(rho.shape)
        if rho > 0:
            theta = n_angle
        else:
            theta = n_angle + math.pi
            rho = -rho

        # xmin = p0[0] - abs(L/2 * math.sin(theta))
        # xmax = p0[0] + abs(L/2 * math.sin(theta))
        polar = polarity.polarity(output_regions[i], gx)
        rho = rho[0, 0]
        theta = theta[0]
        ymin = p0[1] - abs(L / 2 * math.cos(theta))
        ymin = ymin[0][0, 0]
        if ymin <= 0:
            ymin = 0
        # print(type(ymin))
        ymax = p0[1] + abs(L / 2 * math.cos(theta))
        ymax = ymax[0][0, 0]
        if ymax >= 299:
            ymax = 299
        segment_functions[i] = np.array([[rho], [theta], [ymin], [ymax], [polar]])
        # Visualization2: plot line segments detected
        # x = np.arange(xmin, xmax + 0.1, 0.1)
        # y = (rho[0, 0] - x * math.cos(theta)) / math.sin(theta)
        # print(type(ymax)) <class 'numpy.float64'>
        y = np.arange(ymin, ymax + 0.1, 0.1)
        x = (rho - y * math.sin(theta)) / math.cos(theta)
        y_axis = 300 - y  # this instruction is to make two layers match with each
        x_axis = x
        if polar > 0:
            plt.plot(x_axis, y_axis, 'b', lw=1)
        else:
            plt.plot(x_axis, y_axis, 'r', lw=1)
    plt.imshow(imm)
    plt.show()

    # Hyper parameters
    # eps_theta = seg1_theta - seg2_theta, it should be set to a large angle
    eps_theta = math.pi / 4
    eps_rho_base = 5
    eps_dist = 2

    segment_usage = np.zeros((count, 1), dtype=int)
    # combinations is a 'list' of arrays
    combinations = {}
    k = 0

    ydiff = np.zeros((len(segment_functions), 1), dtype=int)
    for i in range(len(segment_functions)):
        ydiff[i] = segment_functions[i][3, 0] - segment_functions[i][2, 0]
        # ymax - ymin

    sort_index = ydiff.argsort(axis=0)

    # calculate edges' elements
    # edges: 1-d np array in ascending order [0, 300)
    cos_sim = np.ones(300)
    eps_sim = 0.70
    edges = np.zeros(1, dtype=int)
    for y0 in range(299):
        if y0 == 0:
            continue
        A = gy[y0, :]
        B = gy[y0 + 1, :]
        dot_product = np.dot(A, B)
        mag_A = np.linalg.norm(A)
        mag_B = np.linalg.norm(B)
        cos_sim[y0] = dot_product / (mag_A * mag_B)
        if cos_sim[y0] < eps_sim < cos_sim[y0 - 1]:
            edges = np.append(edges, np.array([y0]))
            # print(edges)
    edges = np.append(edges, np.array([[299]]))

    for m in range(count):
        ymin = segment_functions[m][2, 0]
        ymax = segment_functions[m][3, 0]
        for n in range(len(edges)-1):
            y0 = edges[n]
            y1 = edges[n+1]
            if y1 > ymin >= y0:
                if ymin+0.5*(ymax-ymin) <= y1:
                    segment_functions[m] = np.concatenate((segment_functions[m], np.array([[y0]])), axis=0)
                else:
                    segment_functions[m] = np.concatenate((segment_functions[m], np.array([[y1]])), axis=0)
        if len(segment_functions[m][:, 0]) == 5:
            print(segment_functions[m][3, 0])
        # print(segment_functions[m].shape)

    for ii in range(count):
        i = int(sort_index[ii])
        if segment_usage[i] == used:
            continue
        # wait_arr = np.zeros((6, 1), dtype=int)
        wait_arr = np.concatenate((segment_functions[i], np.array([[i]])), axis=0)

        segment_usage[i] = used
        for jj in range(ii + 1, count):
            j = int(sort_index[jj])
            if segment_usage[j] == used:
                continue
            # if line_sim == 1, then two segments will be combined together
            if collinearity.collinearity(segment_functions[i], segment_functions[j], eps_dist) == 1:
                # if segment_functions[i][4, 0] == segment_functions[j][4, 0]: # result: two lines
                # if seg1 & seg2 belong to the same segment, then they will be combined together
                wait_arr = np.concatenate((wait_arr, np.concatenate((segment_functions[j],
                                                                     np.array([[j]])), axis=0)), axis=1)
                segment_usage[j] = used
        """
            wait_arr ---    belong to the same segmentation
            cur_comb ---    ndarray type
            cur_ymax ---    int, float...not sure, not ndarray
        """
        # idx = np.argsort(wait_arr.T[:, 2])
        # wait_arr = wait_arr.T[idx].T
        # print(wait_arr.shape) (2, )
        # wait_arr_size = wait_arr.shape[1]
        idx = np.argsort(wait_arr.T[:, 2])
        wait_arr = wait_arr.T[idx].T
        # print(wait_arr.shape)
        wait_arr_size = wait_arr.shape[1]
        tolerance = 10
        cur_comb = np.array([[wait_arr[6, 0]]])
        cur_ymax = wait_arr[3, 0]
        for m in range(1, wait_arr_size):
            # print(segment_functions[wait_arr[5, m]][4:0]) # result: [][]...[]
            # change wait_arr[5, m] into 6 cuz add another row into the array
            print(wait_arr[:, m])
            if cur_ymax + tolerance > wait_arr[2, m]:
                cur_comb = np.concatenate((cur_comb, np.array([[wait_arr[6, m]]])), axis=1)
                if cur_ymax < wait_arr[3, m]:
                    cur_ymax = wait_arr[3, m]
                # cur_ymax = wait_arr[3, m]
            else:
                # combinations[k] = np.concatenate((combinations, cur_comb), axis=1)
                combinations[k] = cur_comb
                print(combinations[k].shape)
                k = k + 1
                # reset cur_comb
                cur_comb = np.array([[wait_arr[6, m]]])
                # if cur_ymax < wait_arr[3, j]:
                #    cur_ymax = wait_arr[3, j]
                cur_ymax = wait_arr[3, m]
        # combinations[k] = np.concatenate((combinations, cur_comb), axis=1)
        combinations[k] = cur_comb
        print(combinations[k].shape)
        k = k + 1

    # Combine some segmentations
    # Hyper-parameter
    # Use simple averaging method, ez to optimize the accuracy
    G_thresh = threshold * 3
    p_num = 0
    n_num = 0
    pos_segments = {}
    neg_segments = {}
    for i in range(len(combinations)):
        # print(i)
        y0 = segment_functions[combinations[i][0, 0]][5, 0]
        polar = segment_functions[combinations[i][0, 0]][4, 0]
        ymin = segment_functions[combinations[i][0, 0]][2, 0]
        # print(ymin)
        ymax = segment_functions[combinations[i][0, 0]][3, 0]
        rho = segment_functions[combinations[i][0, 0]][0, 0]
        # print(rho)
        theta = segment_functions[combinations[i][0, 0]][1, 0]

        # print(theta)
        xmin = (rho - ymin * math.sin(theta)) / math.cos(theta)
        xmax = (rho - ymax * math.sin(theta)) / math.cos(theta)
        # print(1)
        for j in range(len(combinations[i][0, :])):  # this is the bug confusing me a lot...
            idx = combinations[i][0, j]
            new_rho = segment_functions[idx][0, 0]
            new_theta = segment_functions[idx][1, 0]
            new_ymin = segment_functions[idx][2, 0]
            new_ymax = segment_functions[idx][3, 0]
            # print(0)
            if new_ymin < ymin:
                ymin = new_ymin
                xmin = (new_rho - ymin * math.sin(new_theta)) / math.cos(new_theta)
                # print(1)
            if new_ymax > ymax:
                ymax = new_ymax
                xmax = (new_rho - ymax * math.sin(new_theta)) / math.cos(new_theta)
                # print(1)

        theta = math.atan((xmin - xmax) / (ymax - ymin))
        # this should be correct
        rho = xmax * math.cos(theta) + ymax * math.sin(theta)

        if polar > 0:
            pos_segments[p_num] = np.array([[rho], [theta], [ymin], [ymax], [polar], [y0]])
            p_num = p_num + 1
        else:
            neg_segments[n_num] = np.array([[rho], [theta], [ymin], [ymax], [polar], [y0]])
            n_num = n_num + 1

        y = np.arange(ymin, ymax + 0.1, 0.1)
        x = (rho - y * math.sin(theta)) / math.cos(theta)
        # print(rho.shape)
        """visualization 3: plot the grouped line segments with polariry
            if polar > 0
                plot sth
            else:
                plot sth else
        """
        y_axis = 300 - y
        x_axis = x
        if polar > 0:
            plt.plot(x_axis, y_axis, 'b', lw=1)
        else:
            plt.plot(x_axis, y_axis, 'r', lw=1)
    plt.imshow(imm)
    plt.show()
    """
    6th step:
    Matching Line Segments
    """
    K1 = len(pos_segments)
    # debug result: len(pos_segments)=11
    # print(len(pos_segments))
    K2 = len(neg_segments)
    # debug result: len(neg_segments)=11
    # print(K2)
    score_matrix = np.zeros((K1, K2))

    for i in range(K2):
        for j in range(K1):
            score = proximity.proximity(pos_segments[j], neg_segments[i], eps_theta, imm, grayscale=70, ratio=0.35)
            score_matrix[j, i] = score

    order = np.zeros((K1, K2))
    for i in range(K2):
        order[:, i] = np.argsort(score_matrix[:, i])
        order[:, i] = np.flipud(order[:, i])

    pos_matched = np.zeros([K1, 1])
    pos_left = K1
    neg_matched = np.zeros([K2, 1])
    neg_left = K2

    """
    Visualization 4:
    Matched shaded areas with noises.
    """
    pairs = np.zeros((2, 1), dtype=int)
    itr = 0
    n1, n2 = score_matrix.shape
    for j in range(n1):
        for i in range(n2):
            # print(pos_segments[j][4, 0], neg_segments[i][4, 0])
            if score_matrix[j, i] > 0:
                shade.shade(pos_segments[j], neg_segments[i])
                if itr == 0:
                    pairs = np.array([[j], [i]])
                    itr = itr + 1
                else:
                    pairs = np.concatenate((pairs, np.array([[j], [i]])), axis=1)
                    itr = itr + 1
    plt.imshow(imm)
    plt.show()

    """
    7th step:
    Construct stripes using matched line segments
    """
    """
    Visualization 5:
    Expand matched shaded areas with noises to regular geometric shapes.
    """
    # ps:   pairs is a numpy array
    #       pos_segments and neg_segments are two numpy arrays
    #       pos_segments[j] neg_segments[i]
    for k in range(len(pairs[0, :])):
        j = pairs[0, k]
        i = pairs[1, k]
        expand.expand(pos_segments[j], neg_segments[i])
    plt.imshow(imm)
    plt.show()

    """
    8th step:                   
    """

    """
    9th step:
    Compute envelope
    """

    """
    10th step:
    Compute unary potential and binary potential
    """

    """
    11th step:
    Belief propatation, message update
    """


if __name__ == "__main__":
    main()