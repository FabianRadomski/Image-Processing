import cv2
import os
import pandas as pd
import glob
from Localization import plate_detection
from Recognize import segment_and_recognize

"""
In this file, you will define your own CaptureFrame_Process funtion. In this function,
you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
To do:
	1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
	2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
	3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
Inputs:(three)
	1. file_path: video path
	2. sample_frequency: second
	3. save_path: final .csv file path
Output: None
"""
letters = glob.glob("SameSizeLetters/*.bmp")
numbers = glob.glob("SameSizeNumbers/*.bmp")
images = [*letters, *numbers]
templates = [(x[16], cv2.imread(x, cv2.IMREAD_GRAYSCALE)) for x in images]

def CaptureFrame_Process(file_path, sample_frequency, save_path):
	plates = []
	frames = []
	timestamps = []

	cap = cv2.VideoCapture(file_path)
	count = 0

	while cap.isOpened():
		ret, frame = cap.read()
		if ret:
			plate = segment_and_recognize(plate_detection(frame), templates)
			if plate is not None:
				plates.append("".join(plate))
				frames.append(count)
				timestamps.append('')
				print('Frame #' + str(count) + " : " + "".join(plate))
			count += 100
			cap.set(1, count)
		else:
			cap.release()
			break

	measurements = {
		'License plate' : plates,
		'Frame no.' : frames,
		'Timestamp(seconds)' : timestamps
	}
	df = pd.DataFrame(measurements, columns= ['License plate', 'Frame no.', 'Timestamp(seconds)'])
	df.to_csv('Results.csv', index=True, header=True)
	return
