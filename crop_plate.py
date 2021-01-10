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
type(img)

def rotate_both_planes(img, corners):
    width =  max(corners[1][0] - corners[0][0], corners[2][0] - corners[3][0])
    height = max(corners[2][1] - corners[1][1], corners[3][1] - corners[0][1])
    corners = np.float32(corners)
    mappedCorners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    A = np.zeros((12, 9))
    for i in range(4):
        A[i * 3] = [mappedCorners[i][0], mappedCorners[i][1], 1, 0, 0, 0, 0, 0, 0]
        A[i * 3 + 1] = [0, 0, 0, mappedCorners[i][0], mappedCorners[i][1], 1, 0, 0, 0]
        A[i * 3 + 2] = [0, 0, 0, 0, 0, 0, mappedCorners[i][0], mappedCorners[i][1], 1]

    b = np.zeros(12)
    for i in range(4):
        b[i * 3] = corners[i][0]
        b[i * 3 + 1] = corners[i][1]
        b[i * 3 + 2] = 1

    # find the least-squares solution of Ah=b
    # h = vector with value of the transform a,b,c,d,e,f,g,h,i
    h = np.linalg.lstsq(A, b, rcond=None)[0]

    # turn it into a matrix
    M = np.zeros((3, 3))
    for i in range(9):
        M[i // 3][i % 3] = h[i]

    result = np.zeros((height, width , 3), dtype=int)

    for i in range(width):
        for j in range(height):
            cords = np.matmul(M, [i, j, 1])
            x = round(cords[0]/cords[2])
            y = round(cords[1]/cords[2])
            result[j][i] = img[y][x]

    return result

#plate corners
tl = [23, 232]
tr = [161, 244]
br = [159, 277]
bl = [25, 264]
corners = [tl, tr, br, bl]

plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(rotate_both_planes(img, corners))
plt.show()