import matplotlib.pyplot as plt
from Localization import plate_detection
import cv2

frames = [40, 85, 355, 558, 669, 710, 786, 885, 1030]


cap = cv2.VideoCapture('trainingsvideo.avi')
cap.read()


def process_frame(frame, count):
    plate = plate_detection(frame)
    if plate is None:
        print(str(count) + ' is None')
        return
    plt.imshow(plate)
    plt.show()
    plt.title(str(count))
    return plate

i = 1
plates = []
titles = []
cap.set(1, frames[0])
while cap.isOpened():
        ret, frame = cap.read()
        if ret and i < len(frames):
            count = frames[i]
            cap.set(1, count)
            plate = process_frame(frame, count)
            if plate is not None:
                plates.append(plate)
                titles.append(frames[i])
            i += 1
        else:
            cap.release()
            break
