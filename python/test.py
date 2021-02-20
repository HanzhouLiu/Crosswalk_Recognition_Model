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
import matplotlib.path as mpltPath

#polygon = [(5,5),(10,5),(10,10),(5,10)]
polygon = [(5,5),(5,10),(10,10),(10,5)]
width =11
height =11

points1 = [(0,0),[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[8,0],[9,0],[10,0],[11,0], \
          [0,1],[1,1],[2,1],[3,1],[4,1],[5,1],[6,1],[7,1],[8,1],[9,1],[10,1],[11,1],\
          [0,2],[1,2],[2,2],[3,2],[4,2],[5,2],[6,2],[7,2],[8,2],[9,2],[10,2],[11,2],\
          [0,3],[1,3],[2,3],[3,3],[4,3],[5,3],[6,3],[7,3],[8,3],[9,3],[10,3],[11,3],\
          [0,4],[1,4],[2,4],[3,4],[4,4],[5,4],[6,4],[7,4],[8,4],[9,4],[10,4],[11,4],\
          [0,5],[1,5],[2,5],[3,5],[4,5],[5,5],[6,5],[7,5],[8,5],[9,5],[10,5],[11,5],\
          [0,6],[1,6],[2,6],[3,6],[4,6],[5,6],[6,6],[7,6],[8,6],[9,6],[10,6],[11,6],\
          [0,7],[1,7],[2,7],[3,7],[4,7],[5,7],[6,7],[7,7],[8,7],[9,7],[10,7],[11,7],\
          [0,8],[1,8],[2,8],[3,8],[4,8],[5,8],[6,8],[7,8],[8,8],[9,8],[10,8],[11,8],\
          [0,9],[1,9],[2,9],[3,9],[4,9],[5,9],[6,9],[7,9],[8,9],[9,9],[10,9],[11,9],\
          [0,10],[1,10],[2,10],[3,10],[4,10],[5,10],[6,10],[7,10],[8,10],[9,10],[10,10],[11,10],\
          [0,11],[1,11],[2,11],[3,11],[4,11],[5,11],[6,11],[7,11],[8,11],[9,11],[10,11],[11,11]]
points = [(0, 0), (1, 0)]
path = mpltPath.Path(polygon)
inside = path.contains_points(points)
print(inside)

print(points1[0])

list_0 = []
for i in range(3):
    sublist_0 = [0, i]
    list_0.append(sublist_0)
print(list_0)
print(type(list_0[0]))

print("test 10*************************************************")
q = np.array([1.1, 1], dtype=np.int32)
print(q)
