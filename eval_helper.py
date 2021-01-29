from CaptureFrame_Process import CaptureFrame_Process
from multiprocessing import freeze_support


if __name__ == '__main__':
    freeze_support()
    CaptureFrame_Process('trainingsvideo.avi', 1, 'Output.csv')

