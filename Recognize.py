import cv2
import numpy as np
import time
from Localization import dfs

start_time = time.time()

"""
In this file, you will define your own segment_and_recognize function.
To do:
	1. Segment the plates character by character
	2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
	3. Recognize the character by comparing the distances
Inputs:(One)
	1. plate_imgs: cropped plate images by Localization.plate_detection function
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Outputs:(One)
	1. recognized_plates: recognized plate characters
	type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
Hints:
	You may need to define other functions.
"""

def match(letter_box, templates):
    boxH = len(letter_box)
    boxW = len(letter_box[0])

    max_score = 0
    best_match = 'A'

    for string, im in templates:
        score = 0

        tempH = len(im)
        tempW = len(im[0])

        # resize and threshold the template
        res = cv2.resize(im, None, fx=boxH / tempH, fy=boxH /
                         tempH, interpolation=cv2.INTER_LINEAR)
        ret, res = cv2.threshold(res, 5, 1, cv2.THRESH_BINARY)

        # res = resized im so we need to update width and height
        tempH = len(res)
        tempW = len(res[0])

        minW = min(boxW, tempW)
        minH = min(boxH, tempH)
        # all pixels - pixels that were not the same = pixels that were the same
        xor = cv2.bitwise_xor(res[:minH, :minW], letter_box[:minH, :minW])
        score = minW*minH - cv2.countNonZero(xor)

        if score > max_score:
            max_score = score
            best_match = string

    return best_match


def gaussianBlur(img, size):
    kernel = cv2.getGaussianKernel(size, 0)
    return cv2.sepFilter2D(img, -1, kernel, kernel)


def hyphenate(chars, bbs):
    if len(chars) < 6:
        return chars
    diffs = np.zeros(len(chars) - 1, dtype=int)
    for i in range(len(chars) - 1):
        diffs[i] = bbs[i + 1][3] - bbs[i][1]
    biggest_gaps = np.argsort(diffs)[::-1] + 1
    first_pos = biggest_gaps[0]
    second_pos = None
    if np.abs(first_pos - biggest_gaps[1]) > 1:
        second_pos = biggest_gaps[1]
    elif np.abs(first_pos - biggest_gaps[2]) > 1:
        second_pos = biggest_gaps[2]
    else:
        second_pos = biggest_gaps[3]

    chars.insert(max(first_pos, second_pos), '-')
    chars.insert(min(first_pos, second_pos), '-')

    return chars


gaussian_size = 3
background_threshold = 94


def segment_and_recognize(plate_imgs, templates):
    if plate_imgs is None:
        return None

    plate_characters = []
    bbs = []

    gray = cv2.cvtColor(plate_imgs, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], (0, 256)).reshape((256))
    sums = [x*hist[x] for x in range(256)]

    t = 100
    epst = 0.1
    while 1:
        sum_lower = np.sum(hist[:round(t) + 1])
        sum_upper = np.sum(hist[round(t)+1:])
        if sum_lower==0 or sum_upper==0:
            break
        mL = np.sum(sums[:round(t)+1])/sum_lower
        mH = np.sum(sums[round(t)+1:])/sum_upper
        t_new = (mL + mH) / 2

        if abs(t - t_new) < epst:
            break
        t = t_new

    ret, gray = cv2.threshold(gray, t, 1, cv2.THRESH_BINARY_INV)
    visited = np.zeros(gray.shape)

    height, width = gray.shape

    for x in range(plate_imgs.shape[1]):
        for y in range(plate_imgs.shape[0]):
            if visited[y][x] != 0:
                continue
            if gray[y][x] == 0:
                continue

            dfs_map, extremas = dfs(gray, [x, y])
            visited = np.logical_or(dfs_map, visited)

            if extremas[2] - extremas[0] < 0.5*height or extremas[1] - extremas[3] < 0.01*width:
                continue
            if extremas[2] - extremas[0] > 0.95*height or extremas[1] - extremas[3] > 0.16*width:
                continue
            area = np.sum(gray[extremas[0]:extremas[2] + 1, extremas[3]:extremas[1]])/(width*height)
            if area<0.02:
                continue

            bbs.append(extremas)
            plate_characters.append(match(gray[extremas[0]:extremas[2] + 1, extremas[3]:extremas[1]], templates))

    plate_characters = hyphenate(plate_characters, bbs)

    if len(plate_characters) == 0:
        return None

    return plate_characters