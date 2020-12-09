import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
sys.setrecursionlimit(310000)

"""
In this file, you need to define plate_detection function.
To do:
	1. Localize the plates and crop the plates
	2. Adjust the cropped plate images
Inputs:(One)
	1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
	type: Numpy array (imread by OpenCV package)
Outputs:(One)
	1. plate_imgs: cropped and adjusted plate images
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Hints:
	1. You may need to define other functions, such as crop and adjust function
	2. You may need to define two ways for localizing plates(yellow or other colors)
"""

img_nums = [7, 8]  # 3, 4, 5, 7, 8, 10, 12, 13, 14, 17, 20]
f, axarr = plt.subplots(nrows=1, ncols=len(img_nums))


def plate_detection(image):
    # Replace the below lines with your code.
    plate_imgs = image
    return plate_imgs


def filter_yellow(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][0] < 15 or img[i][j][0] > 35 or img[i][j][2] < 100 or img[i][j][1] < 50:
                img[i][j] = [0, 0, 0]
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img


def bgrToRgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def dfs_start(img, point):
    visited = np.zeros(img.shape, dtype=np.ubyte)
    queue = [point]
    while len(queue) > 0:
        point = queue.pop(0)
        x = point[0]
        y = point[1]
        if visited[y][x]:
            continue
        visited[y][x] = 1
        if x > 0 and img[y][x-1] > 0 and visited[y][x-1] == 0:
            queue.append([x-1, y])
        if x < img.shape[1] - 1 and img[y][x+1] > 0 and visited[y][x+1] == 0:
            queue.append([x+1, y])
        if y > 0 and img[y-1][x] > 0 and visited[y-1][x] == 0:
            queue.append([x, y-1])
        if y < img.shape[0] - 1 and img[y+1][x] > 0 and visited[y+1][x] == 0:
            queue.append([x, y+1])

    return visited

def calculateArea(box):
    area = np.abs((box.lb[0] * box.lt[1] - box.lb[1] * box.lt[0] + \
                   box.lt[0] * box.rt[1] - box.lt[1] * box.rt[0] + \
                   box.rt[0] * box.rb[1] - box.rt[1] * box.rb[0] + \
                   box.rb[0] * box.lb[1] + box.rb[1] * box.lb[0]) / 2)
    return area


class BoundingBox:
    def __init__(self, lt, lb, rt, rb):
        self.lt = lt
        self.lb = lb
        self.rt = rt
        self.rb = rb

    def contains(self, point):
        if point[0] >= min(self.lt[0], self.lb[0]) and point[0] <= max(self.rt[0], self.rb[0]):
            if point[1] >= min(self.lb[1], self.rb[1]) and point[1] <= max(self.rt[1], self.lt[1]):
                return True
        return False


def bbFromMap(visited):
    top_y = np.zeros((visited.shape[1], ), dtype=np.uint)
    for x in range(visited.shape[1]):
        for y in range(visited.shape[0]):
            if visited[y][x] == 1:
                top_y[x] = y
                if y == 0:
                    top_y[x] = 1
                break

    bottom_y = np.zeros((visited.shape[1], ), dtype=np.uint)
    for x in range(visited.shape[1]):
        for y in reversed(range(visited.shape[0])):
            if visited[y][x] == 1:
                bottom_y[x] = y
                if y == 0:
                    bottom_y[x] = 1
                break

    epsilon = 4

    top_edge_y = np.average(top_y[np.where(top_y > 0)])
    # find left top
    lt = [0, 0]
    for i, y in enumerate(top_y):
        if y == 0:
            continue
        if np.abs(y - top_edge_y) < epsilon:
            lt[0] = i
            lt[1] = y
            break
    # find right top
    rt = [0, 0]
    for i, y in reversed(list(enumerate(top_y))):
        if y == 0:
              continue
        if np.abs(y - top_edge_y) < epsilon:
            rt[0] = i
            rt[1] = y
            break

    bottom_edge_y = np.average(bottom_y[np.where(bottom_y > 0)])
    # find left bottom
    lb = [0, 0]
    for i, y in enumerate(bottom_y):
        if y == 0:
            continue
        if np.abs(y - bottom_edge_y) < epsilon:
            lb[0] = i
            lb[1] = y
            break
    # find right bottom
    rb = [0, 0]
    for i, y in reversed(list(enumerate(bottom_y))):
        if y == 0:
            continue
        if np.abs(y - bottom_edge_y) < epsilon:
            rb[0] = i
            rb[1] = y
            break

    bb = BoundingBox(lt, lb, rt, rb)
    return bb


ind = 0
for i in img_nums:
    cap = cv2.VideoCapture("TrainingSet/Categorie I/Video" + str(i) + "_2.avi")
    ret, frame = cap.read()
    yellow = filter_yellow(frame)

    kernel1 = np.ones((21, 21), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)
    yellow = cv2.erode(yellow, kernel2, iterations=1)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel2)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel1)
    gray = cv2.cvtColor(yellow, cv2.COLOR_BGR2GRAY)

    boxes = []
    m = np.zeros(gray.shape)

    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            found = False
            if gray[y][x] == 0:
                continue
            # for box in boxes:
            #     if box.contains([x, y]):
            #         found = True
            #         break
            if found:
                continue
            if m[y][x]:
                continue
            d = dfs_start(gray, [x, y])
            m = np.logical_or(m, d)
            boxes.append(bbFromMap(d))

    for box in boxes:
        cv2.line(gray, tuple(box.lt), tuple(box.rt), 30, 1)
        cv2.line(gray, tuple(box.rt), tuple(box.rb), 30, 1)
        cv2.line(gray, tuple(box.rb), tuple(box.lb), 30, 1)
        cv2.line(gray, tuple(box.lb), tuple(box.lt), 30, 1)
        #print(calculateArea(box))
    boxes = sorted(boxes, key=lambda box: calculateArea(box), reverse=True)
    for box in boxes:
        print(calculateArea(box))
        print(box.lb[0])
        print(box.lb[1])
        print(" ")
    axarr[ind].imshow(gray)
    #filter = cv2.GaussianBlur(frame, (5, 5), 0)
    #filter = cv2.bilateralFilter(frame, 23, 10, 150)
    #gray = cv2.cvtColor(filter, cv2.COLOR_BGR2GRAY)
    #filter = cv2.medianBlur(filter, 3)

    #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    #filter = cv2.filter2D(filter, -1, kernel)

    #edged = cv2.Canny(filter, 250, 300)
    ind += 1

plt.show()
