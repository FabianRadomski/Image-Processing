import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
import time
start_time = time.time()

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

img_nums = [17,3, 4]#, 5, 7, 8, 10, 13, 14, 17, 20]
#f, axarr = plt.subplots(nrows=1, ncols=len(img_nums))




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

    max_y = 0
    min_y = 99999
    max_x = 0
    min_x = 99999

    while len(queue) > 0:
        point = queue.pop(0)
        x = point[0]
        y = point[1]

        max_y = max(max_y, y)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        min_x = min(min_x, x)

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

    return (visited, [min_y, max_x, max_y, min_x])

def calculateArea(box):
    area = np.abs((box.lb[0] * box.lt[1] - box.lb[1] * box.lt[0] + \
                   box.lt[0] * box.rt[1] - box.lt[1] * box.rt[0] + \
                   box.rt[0] * box.rb[1] - box.rt[1] * box.rb[0] + \
                   box.rb[0] * box.lb[1] - box.rb[1] * box.lb[0]) / 2)
    return area

def calculateAspectRatio(box):
    edge1 = np.sqrt((int(box.lb[0])-int(box.lt[0]))**2+(int(box.lb[1])-int(box.lt[1]))**2)
    edge2 = np.sqrt((int(box.rb[0])-int(box.lb[0]))**2+(int(box.rb[1])-int(box.lb[1]))**2)
    if edge1==0:
        return -1
    aspect_ratio = edge2/edge1
    return aspect_ratio

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

    epsilon = 7

    top_edge_y = np.median(top_y[np.where(top_y > 0)])
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

    bottom_edge_y = np.median(bottom_y[np.where(bottom_y > 0)])
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
# [tl, tr, br, bl]
def rotate_both_planes(img, box):
    width =  max(box.rt[0] - box.lt[0], box.rb[0] - box.lb[0])
    height = max(box.rb[1] - box.rb[1], box.lb[1] - box.lt[1])
    corners = np.float32([box.lt, box.rt, box.rb, box.lb])
    mappedCorners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    M = cv2.getPerspectiveTransform(corners, mappedCorners)

    return cv2.warpPerspective(img, M, (width, height))

def evaluateBoxes(groundTruthBox, localizedBox):
    sum = 0
    sum += np.sqrt((groundTruthBox.lt[0]-localizedBox.lt[0])**2+(groundTruthBox.lt[1]-localizedBox.lt[1])**2)
    sum += np.sqrt((groundTruthBox.rt[0]-localizedBox.rt[0])**2+(groundTruthBox.rt[1]-localizedBox.rt[1])**2)
    sum += np.sqrt((groundTruthBox.lb[0]-localizedBox.lb[0])**2+(groundTruthBox.lb[1]-localizedBox.lb[1])**2)
    sum += np.sqrt((groundTruthBox.rb[0]-localizedBox.rb[0])**2+(groundTruthBox.rb[1]-localizedBox.rb[1])**2)
    return sum / 4

def generate_horizontal_line(points, startX):
    left_var = np.var(points[:20])
    rigth_var = np.var(points[-20:])

    if left_var < rigth_var:
        points = points[:len(points)-20]
    else:
        points = points[20:]
        startX += 20

    x = np.arange(startX, startX + len(points), dtype=int)
    A = np.vstack([x, np.ones(len(x))]).T

    a, b = np.linalg.lstsq(A, points, rcond=None)[0]

    return (a, b)

def generate_vertical_line(points, startY):
    top_var = np.var(points[:15])
    bottom_var = np.var(points[-15:])

    if top_var < bottom_var:
        points = points[:len(points)-15]
    else:
        points = points[15:]
        startY += 15

    y = np.arange(startY, startY + len(points), dtype=int)
    A = np.vstack([y, np.ones(len(y))]).T

    a, b = np.linalg.lstsq(A, points, rcond=None)[0]

    return (a, b)

# s1 - pair representing the origin of v1
# v1 - pair representing vector v1
# s2 - pair representing the origin of v2
# v2 - pair representing vector v2
# s1+v1*t = s2+v2*u
# v1*t+(-v2)*u = s2-s1
# x[0] holds t and x[1] holds u
# returns a tuple (x,y) = the intersection point
def intersection(s1, v1, s2, v2):
    a = np.array([[v1[0], -v2[0]], [v1[1], -v2[1]]])
    b = np.array([s2[0]-s1[0], s2[1]-s1[1]])
    x = np.linalg.solve(a, b)
    result = (x[0]*v1[0]+s1[0], x[0]*v1[1]+s1[1])
    return result


def find_bounding_lines(dfs_map, extremas):
    img = dfs_map[extremas[0]:extremas[2]+1, extremas[3]:extremas[1]+1]
    plt.imshow(img)

    # Find top line
    top_line_points = np.zeros(extremas[1] - extremas[3] + 1, dtype=int)
    for i in range(len(top_line_points)):
        top_line_points[i] = extremas[0] + np.min(np.where(img[:, i] != 0))

    top_line = generate_horizontal_line(top_line_points, extremas[3])

    # Find bottom line
    bottom_line_points = np.zeros(extremas[1] - extremas[3] + 1, dtype=int)
    for i in range(len(bottom_line_points)):
        bottom_line_points[i] = extremas[0] + np.max(np.where(img[:, i] != 0))

    bottom_line = generate_horizontal_line(bottom_line_points, extremas[3]) 

    # Find left line
    left_line_points = np.zeros(extremas[2] - extremas[0] + 1, dtype=int)
    for i in range(len(left_line_points)):
        left_line_points[i] = extremas[3] + np.min(np.where(img[i, :] != 0))

    left_line = generate_vertical_line(left_line_points, extremas[0])
    # Find right line
    right_line_points = np.zeros(extremas[2] - extremas[0] + 1, dtype=int)
    for i in range(len(right_line_points)):
        right_line_points[i] = extremas[3] + np.max(np.where(img[i, :] != 0))
    right_line = generate_vertical_line(right_line_points, extremas[0])
    

    dis = dfs_map.astype(float)
    for x in range(extremas[3], extremas[1] + 1):
        dis[round(x * top_line[0] + top_line[1])][x] = 0.5
    for x in range(extremas[3], extremas[1] + 1):
        dis[round(x * bottom_line[0] + bottom_line[1])][x] = 0.5
    for y in range(extremas[0], extremas[2] + 1):
        dis[y][round(y * left_line[0] + left_line[1])] = 0.5
    for y in range(extremas[0], extremas[2] + 1):
        dis[y][round(y * right_line[0] + right_line[1])] = 0.5
    plt.imshow(dis)
    

def plate_detection(frame):
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
            dfs_map, extremas = dfs_start(gray, [x, y])
            m = np.logical_or(m, dfs_map)

            # Reject if too small
            if extremas[2] - extremas[0] < 20 or extremas[1] - extremas[3] < 100:
                continue

            find_bounding_lines(dfs_map, extremas)
            # boxes.append(bbFromMap(d))

    # for box in boxes:
    #     cv2.line(gray, tuple(box.lt), tuple(box.rt), 30, 1)
    #     cv2.line(gray, tuple(box.rt), tuple(box.rb), 30, 1)
    #     cv2.line(gray, tuple(box.rb), tuple(box.lb), 30, 1)
    #     cv2.line(gray, tuple(box.lb), tuple(box.lt), 30, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.logical_and(dfs_map, gray)
    boxes = sorted(boxes, key=lambda box: calculateArea(box), reverse=True)
    best_box = boxes[0]
    for box in boxes[:3]:
        ar = calculateAspectRatio(box)
        if ar>4.3 and ar<7:
            best_box = box
            break
    return rotate_both_planes(frame, best_box)

bb_ev = [
    BoundingBox([352,254],[353,282],[484,250],[484,275]),
    BoundingBox([247,287],[244,314],[388,290],[386,317]),
    BoundingBox([276,165],[275,194],[420,173],[420,202]),
    BoundingBox([305,341],[306,367],[431,338],[430,362]),
    BoundingBox([349,232],[349,260],[497,226],[494,252]),
    BoundingBox([215,315],[215,359],[435,320],[431,364]),
    BoundingBox([285,243],[286,266],[409,243],[410,267]),
    BoundingBox([275,340],[277,382],[508,321],[511,378]),
    BoundingBox([133,325],[131,429],[625,330],[629,429]),
    BoundingBox([266,328],[263,359],[408,345],[406,378])
]

ind = 0
for i in img_nums:
    cap = cv2.VideoCapture("TrainingSet/Categorie I/Video" + str(i) + "_2.avi")
    ret, frame = cap.read()
    plate_detection(frame)
    #axarr[ind].imshow(plate_detection(frame))
    ind += 1
print("--- %s seconds ---" % str((time.time() - start_time) / len(img_nums)))

plt.show()

