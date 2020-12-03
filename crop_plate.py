import cv2
import numpy as np
import matplotlib.pyplot as plt

def crop_img(img, p1, p2):
    top_x = max(p1[0], p2[0]) + 1
    bottom_x = min(p1[0], p2[0])

    top_y = max(p1[1], p2[1]) + 1
    bottom_y = min(p1[1], p2[1])
    return img[bottom_x:top_x, bottom_y:top_y]


img = cv2.imread('krzywa.png', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rotate_both_planes(img, corners):
    width =  max(corners[1][0] - corners[0][0], corners[2][0] - corners[3][0])
    height = max(corners[2][1] - corners[1][1], corners[3][1] - corners[0][1])
    print(width, height)
    corners = np.float32(corners)
    mappedCorners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    M = cv2.getPerspectiveTransform(corners, mappedCorners)

    return cv2.warpPerspective(img, M, (width, height))

#plate corners
tl = [23, 232]
tr = [161, 244]
br = [159, 277]
bl = [25, 264]
corners = [tl, tr, br, bl]

plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(rotate_both_planes(img, corners))
plt.show()