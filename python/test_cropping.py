"""
For testing cropping and mask functions use. 
"""


import numpy as np
import cv2

img = cv2.imread("../Samples/image003.jpg")

# convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# define polygon points
points = np.array( [[[200,0],[230,0],[230,30],[200,30]]], dtype=np.int32 )

# draw polygon on input to visualize
img_poly = img.copy()
cv2.polylines(img_poly, [points], True, (0,0,255), 1)

# create mask for polygon
mask = np.zeros_like(gray)
cv2.fillPoly(mask,[points],(255))

# get color values in gray image corresponding to where mask is white
values = gray[np.where(mask == 255)]

# count number of white values
count = 0
for value in values:
    if value == 255:
        count = count + 1
print("count =",count)

if count > 5:
    result = img.copy()
    result[mask==255] = (0,0,0)
else:        
    result = img


# save results
cv2.imwrite('../data/mask_test/barn_polygon.png', img_poly)
cv2.imwrite('../data/mask_test/barn_mask.png', mask)
cv2.imwrite('../data/mask_test/barn_poly_result.png', result)

cv2.imshow('../data/mask_test/barn_poly', img_poly)
cv2.imshow('../data/mask_test/barn_mask', mask)
cv2.imshow('../data/mask_test/barn_result', result)
cv2.waitKey()