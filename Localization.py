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

import matplotlib.pyplot as plt
f, axarr = plt.subplots(3)


cap = cv2.VideoCapture("TrainingSet/Categorie I/Video30_2.avi")
ret, frame = cap.read()

hsl = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

for i in range(frame.shape[0]):
    for j in range(frame.shape[1]):
        if hsl[i][j][0] < 15 or hsl[i][j][0] > 25 or hsl[i][j][2] < 127:
            frame[i][j] = [255, 255, 255]


filter = cv2.GaussianBlur(frame, (5, 5), 0)
#filter = cv2.bilateralFilter(frame, 23, 10, 150)
#gray = cv2.cvtColor(filter, cv2.COLOR_BGR2GRAY)
#filter = cv2.medianBlur(filter, 3)
print(hsl[290, 250])
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
filter = cv2.filter2D(filter, -1, kernel)
axarr[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

edged = cv2.Canny(filter, 250, 300)
axarr[0].imshow(cv2.cvtColor(filter, cv2.COLOR_BGR2RGB))
axarr[1].imshow(edged)

contours, ne  = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[0:50]

cntrs = []
m = 0
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.1 * peri, True)
    if len(approx) == 4:
        cntrs.append(c)
        #break

axarr[2].imshow(cv2.drawContours(frame, cntrs, -1, (255, 0, 0), 3))
plt.show()