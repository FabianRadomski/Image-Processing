import cv2
import numpy as np

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
def plate_detection(image):
    #Replace the below lines with your code.
    plate_imgs = image
    return plate_imgs

cap = cv2.VideoCapture("Video7_2.avi")
ret, frame = cap.read()

import matplotlib.pyplot as plt

#blur = cv2.GaussianBlur(frame, (7,7), 0)
filter = cv2.bilateralFilter(frame, 17, 50, 150)
gray = cv2.cvtColor(filter, cv2.COLOR_BGR2GRAY)


edged = cv2.Canny(gray, 10, 170)

contours, ne  = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

cntrs = []
m = 0
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.1 * peri, True)
    m = max(len(approx), m)
    if len(approx) == 4:
        cntrs.append(c)
        break

print(m)

mask = np.zeros(gray.shape, np.uint8)
# img_cntrd = cv2.drawContours(mask, [cntrs], 0, 255, -1)
# img_cntrd = cv2.bitwise_and(frame, frame, mask=mask)
plt.imshow(cv2.drawContours(frame, contours, -1, (255, 0, 0), 3))
plt.show()
