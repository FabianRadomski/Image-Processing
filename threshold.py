from __future__ import print_function
import cv2 as cv
import argparse

import numpy as np

from Localization import plate_detection

max_value = 255
max_value_H = 360 // 2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
alpha = 1
beta = 1
window_capture_name = 'Video Capture'
window_detection_name = 'HSV threshold'
detect_show = "Object detection"
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
alpha_name = 'alpha'
beta_name = 'beta'


def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H - 1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)


def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H + 1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)


def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S - 1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)


def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S + 1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)


def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V - 1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)


def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V + 1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)

def alpha_trackbar(val):
    global alpha
    alpha = val
    cv.setTrackbarPos(alpha_name, window_detection_name, alpha)

def beta_trackbar(val):
    global beta
    beta = val
    cv.setTrackbarPos(beta_name, window_detection_name, beta)

parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()
# cap = cv.VideoCapture(args.camera)
img_nums = [4, 5, 7, 8, 10, 13, 14, 17, 20]
i = 0
cap = cv.VideoCapture("TrainingSet/Categorie I/Video" + str(img_nums[i]) + "_2.avi")
ret, frame = cap.read()
frame = plate_detection(frame)

cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.resizeWindow(window_detection_name, 300, 400)
cv.namedWindow(detect_show)
cv.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)
cv.createTrackbar(alpha_name, window_detection_name, alpha, 1000, alpha_trackbar)
cv.createTrackbar(beta_name, window_detection_name, beta, 500, beta_trackbar)


# odpalasz se normalnie i ustawiasz alpha na 100 i beta na 250 bo chciałem więcej stepów mieć a nie umiem w trackbary xdd
# n to poprzedni frame a m następny no i HSV czaisz, ogólnie to średnio pomaga ten kontrast xd ale popróbuj
while True:


    # ret, frame = cap.read()
    if frame is None:
        break
    scaledframe = cv.convertScaleAbs(frame, alpha=alpha/100, beta=beta-250)
    frame_HSV = cv.cvtColor(scaledframe, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

    cv.imshow(window_capture_name, scaledframe)
    cv.imshow(detect_show, frame_threshold)

    key = cv.waitKey(30)
    if key == ord('n'):
        if i==0:
            continue
        i-=1
        cap = cv.VideoCapture("TrainingSet/Categorie I/Video" + str(img_nums[i]) + "_2.avi")
        ret, frame = cap.read()
        frame = plate_detection(frame)

    if key == ord('m'):
        if i==len(img_nums)-1:
            continue
        i += 1
        cap = cv.VideoCapture("TrainingSet/Categorie I/Video" + str(img_nums[i]) + "_2.avi")
        ret, frame = cap.read()
        frame = plate_detection(frame)


    if key == ord('q') or key == 27:
        break