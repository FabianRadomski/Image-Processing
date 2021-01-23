import sys
from typing import Text
import cv2
import pandas as pd
import glob
from Localization import plate_detection
from Recognize import segment_and_recognize
from multiprocessing import JoinableQueue, cpu_count
from threading import Thread, Event
from time import sleep, time
from operator import itemgetter

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

def localize_and_recognize(frame):
	return segment_and_recognize(plate_detection(frame), templates)

def process(q, results, exit_flag):
	if exit_flag.is_set():
		sys.exit(0)

	while not q.empty():
		frame, frame_count, timestamp = q.get()
		plate = localize_and_recognize(frame)
		if plate is not None:
			if len(plate) == 8:
				results.append(["".join(plate), frame_count, timestamp])
		q.task_done()
	return True

def count_mismatches(plate1, plate2):
	mismatches = 0
	for a, b in zip(plate1, plate2):
		if a != b:
			mismatches += 1
	return mismatches

def most_frequent(List): 
    dict = {} 
    count, itm = 0, '' 
    for item in reversed(List): 
        dict[item] = dict.get(item, 0) + 1
        if dict[item] >= count : 
            count, itm = dict[item], item 
    return(itm) 

def majority_vote(results):
	res_plate = ''
	for i in range(8):
		chars = []
		for plate in results:
			chars.append(plate[i])
		res_plate += most_frequent(chars)
	return res_plate

def synthesize(results, fps):
	synthesized_results = []
	
	cur_start = 0

	
	for i in range(len(results) - 1):
		diff = count_mismatches(results[i][0], results[i + 1][0])
		if diff < 3:
			continue
		
		elem = 0
		sum = 0
		for j in range(2, 7):
			if i + j >= len(results):
				continue
			elem += 1
			sum += count_mismatches(results[i][0], results[i + j][0])
		avg = 0
		if elem != 0:
			avg = sum / elem
		if avg > 3:
			if i - cur_start > 6:
				frame = (results[cur_start][1] + results[i][1]) // 2
				plates = []
				for p in range(cur_start, i + 1):
					plates.append(results[p][0])
				synthesized_results.append((majority_vote(plates), frame, ("%.2f" % (frame / fps))))
			cur_start = i + 1
	if len(results) - cur_start > 6:
		frame = (results[cur_start][1] + results[len(results) - 1][1]) // 2
		plates = []
		for p in range(cur_start, len(results)):
			plates.append(results[p][0])
		synthesized_results.append((majority_vote(plates), frame, ("%.2f" % (frame / fps))))
	return synthesized_results

def CaptureFrame_Process(file_path, sample_frequency, save_path):
	results = []
	cap = cv2.VideoCapture(file_path)
	fps = cap.get(cv2.CAP_PROP_FPS)
	count = 0

	num_threads = 16
	q = JoinableQueue(maxsize=0)
	
	print("Number of threads " + str(num_threads))

	while cap.isOpened():
		ret, frame = cap.read()
		if ret and count < 1541:
			q.put((frame, count, ("%.2f" % (count / fps))))
			count += sample_frequency
			cap.set(1, count)
		else:
			cap.release()
			break
	

	exit_flag = Event()
	exit_flag.clear()

	#print("Starting processing plates")
	epoch_start = time()
	threads = []
	for i in range(num_threads):
		worker = Thread(target=process, args=(q, results, exit_flag))
		worker.daemon = True
		worker.start()
		threads.append(worker)
	q.join()
	exit_flag.set()

	sleep(2)

	epoch_end = time()

	#print("Time: " + str(epoch_end - epoch_start))
	results = sorted(results, key=itemgetter(1))
	
	df = pd.DataFrame(synthesize(results, fps), columns= ['License plate', 'Frame no.', 'Timestamp(seconds)'])
	df.to_csv(save_path, index=False, header=True)
	return
