import numpy as np

from ultils import *
from setting import win_size
from sliding_window1 import *
import cv2
params=load_classifier("ver1.p", path='/Users/datle/Desktop/license_plate_detection/train_vehicle_detection')
print(params)
image = cv2.imread("/Users/datle/Desktop/Official_license_plate/Training_vehicle_detection/result/middle_close.jpeg")
(winH, winW,ch) = params['size_of_pic_train']
scaler= params['scaler']
model=params['svc']
import time

