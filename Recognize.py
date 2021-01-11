import cv2
import numpy as np
import os

import matplotlib.pyplot as plt
from Localization import plate_detection, dfs

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

img_nums = [4] #4, 5, 7, 8, 10, 13, 14, 17, 20
#f, axarr = plt.subplots(nrows=1, ncols=len(img_nums))

def match(letter_box):
	#TODO
	return 'A'


guassian_size = 3
background_threshold = 90
def segment_and_recognize(plate_imgs):
	plate_characters = []

	gray = cv2.cvtColor(np.float32(plate_imgs), cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (guassian_size, guassian_size) , 0)
	gray = gray.astype(int)
	for y in range(plate_imgs.shape[0]):
		for x in range(plate_imgs.shape[1]):
			if gray[y][x] < background_threshold:
				gray[y][x] = 1
			else:
				gray[y][x] = 0

	visited = np.zeros(gray.shape)

	for y in range(plate_imgs.shape[0]):
		for x in range(plate_imgs.shape[1]):
			if visited[y][x] == 1:
				continue
			if gray[y][x] == 0:
				continue

			dfs_map, extremas = dfs(gray, [x, y])
			visited = np.logical_or(visited, dfs_map)

			# reject noise
			if extremas[2] - extremas[0] < 15 or extremas[1] - extremas[3] < 4:
				continue

			plate_characters.append(match(gray[extremas[0]:extremas[2] + 1, extremas[3]:extremas[1]]))
			plt.imshow(gray[extremas[0]:extremas[2] + 1, extremas[3]:extremas[1]])
			plt.show()
	plt.imshow(visited)
	return gray

ind = 0
for i in img_nums:
    cap = cv2.VideoCapture("TrainingSet/Categorie I/Video" + str(i) + "_2.avi")
    ret, frame = cap.read()
    #plt.imshow(segment_and_recognize(plate_detection(frame)))
    #axarr[ind].imshow()
    segment_and_recognize(plate_detection(frame))
    ind += 1

plt.show()