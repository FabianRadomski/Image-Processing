import matplotlib.pyplot as plt
import cv2
import numpy as np
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


def filter_yellow(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (15, 50, 100), (35, 255, 255))

def dfs(img, point):
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


# [tl, tr, br, bl]
def rotate_both_planes(img, corners):
    width = min(img.shape[1], max(int(corners[1][0]) - int(corners[0][0]), int(corners[2][0]) - int(corners[3][0])))
    height = min(img.shape[0], max(int(corners[2][1]) - int(corners[1][1]), int(corners[3][1]) - int(corners[0][1])))
    corners = np.float32(corners)
    mappedCorners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    if height < 1 or width < 1:
        return None

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

    result = np.zeros((height, width, 3), dtype=np.uint8)

    for j in range(height):
        for i in range(width):
            cords = np.matmul(M, [i, j, 1])
            x = np.clip(round(cords[0]/cords[2]), 0, img.shape[1]-1)
            y = np.clip(round(cords[1]/cords[2]), 0, img.shape[0]-1)
            result[j][i] = img[y][x]

    return result

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

def get_middle(arr, radius, startY):
    if len(arr) <= 2 * radius + 1:
        return (arr, startY)
    m = len(arr) // 2
    return (arr[m - radius:m+radius], startY - radius)


def generate_vertical_line(points, startY):
    cut = len(points) // 3

    top_var = np.var(points[:cut])
    bottom_var = np.var(points[-cut:])

    if top_var < bottom_var:
        points = points[:len(points)-cut]
    else:
        points = points[cut:]
        startY += cut


    mid_cut = len(points) // 5

    points, startY = get_middle(points, mid_cut, startY)


    diffs = np.zeros(len(points) - 1)
    for i in range(len(points) - 1):
        diffs[i] = np.abs(points[i] - points[i + 1])

    coords = []
    median_diff = np.quantile(diffs, 0.15)
    if median_diff == 0:
        median_diff = 1

    for i in range(len(diffs)):
        if diffs[i] < median_diff:
            coords.append([startY + i, points[i]])

    coords = np.array(coords)
    y = np.array(coords[:, 0])
    A = np.vstack([y, np.ones(len(y))]).T

    a, b = np.linalg.lstsq(A, coords[:, 1], rcond=None)[0]

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

    vecTop = (1, top_line[0])
    topx = (extremas[1]+extremas[3])/2
    startTop = (topx, top_line[0] * topx + top_line[1])

    vecBot = (1, bottom_line[0])
    startBot = (topx, bottom_line[0] * topx + bottom_line[1])


    vecLeft = (left_line[0], 1)
    lefty = (extremas[0]+extremas[2])/2
    startLeft = (left_line[0] * lefty + left_line[1], lefty)

    vecRight = (right_line[0], 1)
    startRight = (right_line[0] * lefty + right_line[1], lefty)

    leftTop = intersection(startLeft, vecLeft, startTop, vecTop)
    rightTop = intersection(startRight, vecRight, startTop, vecTop)
    rightBot = intersection(startRight, vecRight, startBot, vecBot)
    leftBot = intersection(startLeft, vecLeft, startBot, vecBot)

    corners = np.array([leftTop, rightTop, rightBot, leftBot])

    return corners

    

def plate_detection(frame):
    if frame is None:
        return None
    yellow = filter_yellow(frame)

    kernel1 = np.ones((21, 21), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)

    yellow = cv2.erode(yellow, kernel2, dst=yellow, iterations=1)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel2)
    gray = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel1)

    m = np.zeros(gray.shape)

    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):

            if gray[y][x] == 0:
                continue

            if m[y][x]:
                continue
            dfs_map, extremas = dfs(gray, [x, y])
            m = np.logical_or(m, dfs_map)

            # Reject if too small
            if extremas[2] - extremas[0] < 20 or extremas[1] - extremas[3] < 100:
                continue

            corners = find_bounding_lines(dfs_map, extremas)
            return rotate_both_planes(frame, corners)
    return None
