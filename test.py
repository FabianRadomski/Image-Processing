from Recognize import segment_and_recognize
from Localization import plate_detection
import matplotlib.pyplot as plt
from os import walk
import numpy as np
import cv2


folder = 'Test'
_, _, filenames = next(walk(folder))

names = ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
accuracies = {}
for name in names:
    accuracies[name]=[0, 0]


for img in filenames:
    try:
        img.index('png')
    except ValueError:
        continue

    plate = cv2.imread(folder + '/' + img)
    #plate = plate_detection(img_file)
    if plate is None:
        print(img.split('.')[0] + ' - Localization failed!')
        continue
    result = "".join(segment_and_recognize(plate))
    expected = img.split('.')[0]

    for i,char in enumerate(expected):
        #if result[min(i, len(result)-1)] == char:
        #    accuracies[char][0] += 1
        accuracies[char][1] +=1


    if result is None:
        print(img.split('.')[0] + ' - Failed!')
        continue
    if img.split('.')[0] == result:
        print(img.split('.')[0] + ' - Flawless')
    else:
        print('Expected: ' + img.split('.')[0] + ' Got: ' + result)
print(accuracies)