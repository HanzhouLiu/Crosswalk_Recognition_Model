import numpy as np
import math

img = np.random.rand(20, 30)
img = np.array(img)
[w, h] = img.shape
print(w, h)

zero_padded_img = np.zeros([w + 1, h + 1])

output = np.zeros([img.shape[0], img.shape[1]])
kernel = np.array([[1, -1], [1, -1]])
empty_arr = np.zeros([2, 2])
for row in range(img.shape[0]):
    # Exit Convolution
    for col in range(img.shape[1]):
        empty_arr = img[row:row + 2:1, col:col + 2:1]
        # print(empty_arr, row, col)
        product_arr = kernel * empty_arr
        output[row, col] = sum([sum(i) for i in product_arr])
# print(img)
print("test 1*************************************************")
print(output)

# arr1 = np.array([[1, 2], [3, 4]])
# print(sum(sum(arr1 * arr1, [])))
# print(arr1[:, :] * arr1)
# print(sum(sum([[1, 2], [3, 4]], [])))
# print(sum([sum(i) for i in arr1 * arr1]))

print(img[0:3:1, 0:3:1])
print("test 2*************************************************")

x = 2
print(x * x)

print("test 3*************************************************")

lis = []
for i in range(img.shape[0]):
    lis.append(max(img[i]))
print(max(lis))

print("test 4*************************************************")

arr = np.array([10, -1, 0, 2])
print(arr)
idx = np.where(arr == arr.min())
print(idx)
print(arr[idx])
arr_ones = np.ones((4, 1))
print(arr_ones)

print("test 5*************************************************")

x = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
print(x[1, :, 0])
print(len(x[1, :, 0]))
print(x[0, :, 1].shape)
x_2 = np.zeros((4, 3, 2))
print(x_2)

print("test 6*************************************************")
import matplotlib.pyplot as plt
import numpy as np

# 100 linearly spaced numbers
x = np.linspace(-5, 5, 100)

# the function, which is y = x^2 here
y = x ** 2

# setting the axes at the centre

print("test 7*************************************************")
for i in range(1, 3):
    print(i)

print("test 8*************************************************")
offset = np.array([[-1, -1, -1, 0, 0, 1, 1, 1],
                   [-1, 0, 1, -1, 1, -1, 0, 1]])
region = np.array([[1, -1]]).T
print(np.arange(0, 5, 0.5))

print("test 9*************************************************")
m = np.array([[1], [2], [3]])
print(m.T * m)

print("test 10*************************************************")
import matplotlib.pyplot as plt

# define corner points
x = [1,2,1,0]
y = [2,1,0,1]

# plot
im = plt.fill(x, y)


