import cv2 as cv
import numpy as np
import math_tools as mt
import math
import region_grow
import rectangle_self
import polarity
from PIL import Image
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.cluster import KMeans

# Python Imaging Library


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
    # cv.imshow("img", img)
    # cv.waitKey(0)

    """
    1st step: GaussianBlur
    Filter the image with a Gaussian filter with standard deviation of 0.6/0.8
    Therefore, a Gaussian kernel is used.
    It's done with the function cv2.GaussianBlur()
    """
    img_blur = cv.GaussianBlur(img, sigmaX=0.6/0.8, sigmaY=0.6/0.8, ksize=(3, 3))
    cv.addWeighted(img_blur, 1.5, img, -0.5, 0, img)

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
    # gx = cv.filter2D(img, -1, kernel=x_kernel, anchor=(-1, -1), borderType=cv.BORDER_DEFAULT)
    # gy = cv.filter2D(img, -1, kernel=y_kernel, anchor=(-1, -1))
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
    tau = math.pi/8

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
                if abs(abs(rect_angle) - math.pi / 2) < math.pi / 12:
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
    """:cvar
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
        ymin = p0[1] - abs(L/2 * math.cos(theta))
        ymax = p0[1] + abs(L/2 * math.cos(theta))
        #xmin = p0[0] - abs(L/2 * math.sin(theta))
        #xmax = p0[0] + abs(L/2 * math.sin(theta))
        polar = polarity.polarity(output_regions[i], gy)
        segment_functions[i] = np.array([[rho], [theta], [ymin], [ymax], [polar]])
        # visualization: plot line segments detected
        #x = np.arange(xmin, xmax + 0.1, 0.1)
        #y = (rho[0, 0] - x * math.cos(theta)) / math.sin(theta)
        # print(y)
        y = np.arange(ymin, ymax + 0.1, 0.1)
        x = (rho[0, 0] - y * math.sin(theta)) / math.cos(theta)
        plt.plot(x, y, lw=1)
        # print(i)
    plt.show()
        # print(1)


if __name__ == "__main__":
    main()
