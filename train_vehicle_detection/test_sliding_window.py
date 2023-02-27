import cv2
import time

import numpy as np

from ultils import load_classifier, get_feature_of_image
img= cv2.imread('/Users/datle/Desktop/Official_license_plate/Training_vehicle_detection/result/run_load_data.jpeg', cv2.IMREAD_COLOR)
def sliding_window(img,params, y_start_stop=[None, None], scale=1.5, overlap_rate=0.5):
    if y_start_stop[0] == None or y_start_stop[0] > img.shape[0]:
        y_start_stop[0]=0
    if y_start_stop[1]== None  or y_start_stop[1] > img.shape[0]:
        y_start_stop[1]= img.shape[0]

    img=img[y_start_stop[0]:y_start_stop[1],:,:]
    model= params['svc']
    scaler= params['scaler']
    if scale!=1:
        img=cv2.resize(img, (int(img.shape[1]/scale), int(img.shape[0]/scale)))
    size_of_image=img.shape
    size_of_window=params['size_of_pic_train']
    pixel_skip=size_of_window[0]*overlap_rate
    over_lap=size_of_window[0]/pixel_skip
    number_of_win_per_x= int((size_of_image[1]/size_of_window[1])* over_lap)-1
    number_of_win_per_y= int((size_of_image[0]/size_of_window[0])* over_lap)-1
    bboxs=[]
    for y in range(number_of_win_per_y):
        for x in range(number_of_win_per_x):
            x_pos=int(x*pixel_skip)
            y_pos=int(y*pixel_skip)
            x_pos_end=int(x_pos+size_of_window[1])
            y_pos_end=int(y_pos+size_of_window[0])
            img_crop=img[y_pos:y_pos_end, x_pos:x_pos_end]
            feature= get_feature_of_image(img_crop, orient=params['orient'], pix_per_cell=params['pix_per_cell'],
                                         cell_per_block=params['cell_per_block'],
                                         feature_vector=False, special=False, color_space=params['color_space'])
            feature_scale=scaler.transform(np.array(feature).reshape(1,-1))
            prediction= model.predict(feature_scale)
            if prediction==1:
                bboxs.append([int(x_pos*scale),int(y_pos*scale),int(x_pos*scale+size_of_window[1]),int(y_pos*scale+size_of_window[0])])
                # cv2.rectangle(img,(x_pos,y_pos),(x_pos_end,y_pos_end),(255,0,0))
                # cv2.imshow('r', img)
                # cv2.waitKey(1)
                # time.sleep(0.05)
    return bboxs
def draw(img,box):
    for x in box:
        cv2.rectangle(img, (x[0],x[1]), (x[2],x[3]), (0,0,255), 2)
    return img

params= load_classifier('ver1.p',path= "/Users/datle/Desktop/plate_v3/train_vehicle_detection")
bboxs=sliding_window(img, params, scale=1.0)
img=draw(img, bboxs)
cv2.imshow('r', img)
cv2.waitKey(0)
