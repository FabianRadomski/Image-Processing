from Recognize import segment_and_recognize
from Localization import plate_detection
import matplotlib.pyplot as plt
from os import walk
import numpy as np
import cv2


folder = 'Test'
_, _, filenames = next(walk(folder))

for img in filenames:
    try:
        img.index('png')
    except ValueError:
        continue

    img_file = cv2.imread(folder + '/' + img)
    plate = plate_detection(img_file)
    if plate is None:
        print(img.split('.')[0] + ' - Localization failed!')
        continue
    result = "".join(segment_and_recognize(plate))
    if result is None:
        print(img.split('.')[0] + ' - Failed!')
        continue
    if img.split('.')[0] == result:
        print(img.split('.')[0] + ' - Flawless')
    else:
        print('Expected: ' + img.split('.')[0] + ' Got: ' + result)
