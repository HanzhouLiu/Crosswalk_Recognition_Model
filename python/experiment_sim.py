import cv2 as cv
import numpy as np
import math_tools as mt
import math
import region_grow
import rectangle_self
import polarity
import line_sim
import proximity
import collinearity
import shade
import expand
from PIL import Image
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.cluster import KMeans
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage import data


# Python Imaging Library


def main():
    # Hyper-parameters
    M = 300
    N = 400

    print("zebra-crossing detection algorithm")
    filename = '../Samples/image009.jpg'
    img = cv.imread(filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # white pixel == 255
    # black pixel == 0
    img = cv.resize(gray, (N, M))
    imm = img
    img = 255 - img

    img_blur = cv.GaussianBlur(img, sigmaX=0.6 / 0.8, sigmaY=0.6 / 0.8, ksize=(3, 3))
    cv.addWeighted(img_blur, 1.5, img, -0.5, 0, img)

    # Visualization1: Preprocessed image
    cv.imshow("img", imm)
    cv.waitKey(0)

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
    G = mt.sqrt(gx, gy)
    angle = mt.atan2(-gy, gx)
    cos_sim = np.ones(300)
    eps_sim = 0.70
    for y0 in range(299):
        A = gy[y0, :]
        B = gy[y0 + 1, :]
        dot_product = np.dot(A, B)
        mag_A = np.linalg.norm(A)
        mag_B = np.linalg.norm(B)
        cos_sim[y0] = dot_product / (mag_A * mag_B)
        print('row' + str(y0) + ':' + str(cos_sim[y0]))
        if cos_sim[y0] < eps_sim < cos_sim[y0 - 1]:
            plt.axhline(y=y0, color='r', linestyle='-')
    plt.imshow(imm)
    plt.show()


if __name__ == "__main__":
    main()
