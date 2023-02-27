import cv2
from ultils import *
from setting import win_size
params= load_classifier('ver1.p', path="/Users/datle/Desktop/license_plate_detection/train_vehicle_detection")
test(params=params ,win_size=win_size)
