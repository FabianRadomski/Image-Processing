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

def filter_yellow(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  for i in range(img.shape[0]):
      for j in range(img.shape[1]):
          if img[i][j][0] < 10 or img[i][j][0] > 35 or img[i][j][2] < 100 or img[i][j][1]<40:
              img[i][j] = [0, 0, 0]
  img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
  return img

def bgrToRgb(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


img_nums = [2, 3]#, 4, 5, 7, 8, 10, 12, 13, 14, 17, 20]

#takes brg

f, axarr = plt.subplots(nrows=1, ncols=len(img_nums))
ind = 0
for i in img_nums:
    cap = cv2.VideoCapture("TrainingSet/Categorie I/Video" + str(i) + "_2.avi")
    ret, frame = cap.read()

    yellow = filter_yellow(frame)

    kernel1 = np.ones((20, 20), np.uint8)
    kernel2 = np.ones((2, 2), np.uint8)
    yellow = cv2.erode(yellow, kernel2, iterations=2)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel1)
    axarr[ind].imshow(yellow)

    filter = cv2.GaussianBlur(frame, (5, 5), 0)
    #filter = cv2.bilateralFilter(frame, 23, 10, 150)
    #gray = cv2.cvtColor(filter, cv2.COLOR_BGR2GRAY)
    #filter = cv2.medianBlur(filter, 3)

    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    filter = cv2.filter2D(filter, -1, kernel)

    edged = cv2.Canny(filter, 250, 300)
    ind += 1

plt.show()