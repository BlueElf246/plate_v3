import glob
import os
import random

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, NuSVC
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pickle
from scipy.ndimage import label
import time
import cv2
def load_dataset(name1, name2, num_ex=100):
    car=[]
    for x in name1:
        car+= glob.glob(x)
    non_car=[]
    for y in name2:
        non_car+= glob.glob(y)
    random.shuffle(car)
    random.shuffle(non_car)
    return car[:num_ex], non_car[:num_ex]
def get_feature_of_image(img, orient=9, pix_per_cell=8, cell_per_block=2,
                         feature_vector=True, vis=False,
                         special=True, color_space='RGB'):
    h=[]
    if color_space =='gray':

        h.append(hog(img,  orient, pixels_per_cell=(pix_per_cell,pix_per_cell), cells_per_block=(cell_per_block,cell_per_block),
                     feature_vector=feature_vector, visualize=vis, transform_sqrt=False, block_norm='L2-Hys'))
    else:
        for x in range(3):
            hog_feature= hog(img[:,:,x], orient, pixels_per_cell=(pix_per_cell,pix_per_cell), cells_per_block=(cell_per_block,cell_per_block),
                         feature_vector=feature_vector, visualize=vis, transform_sqrt=False, block_norm='L2-Hys')
            h.append(hog_feature)
    if special==True:
        return h
    return np.concatenate(h)
def change_color_space(img,colorspace):
    if colorspace != 'RGB':
        if colorspace == "YCrCb":
            img=cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if colorspace == 'hls':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        if colorspace == 'yuv':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        if colorspace == 'gray':
            img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def extract_feature(dataset, params):
    dataset_feature=[]
    save=True
    pix_per_cell_1=None
    color_space= params['color_space']
    print(params)
    for x in dataset:
        img=cv2.imread(x, cv2.IMREAD_COLOR)

        img_resized= cv2.resize(img,(params['size_of_window'][0],params['size_of_window'][1]))
        feature=get_feature_of_image(img_resized, orient=params['orient'], pix_per_cell=params['pix_per_cell'], cell_per_block=params['cell_per_block'],
                                     feature_vector=True, special=False, color_space=color_space)
        dataset_feature.append(feature)
    return dataset_feature

def combine(car, non_car):
    X= np.vstack((car,non_car)).astype(np.float32)
    y= np.hstack((np.ones(len(car)), np.zeros(len(non_car))))
    return X,y
def normalize(X):
    sc=StandardScaler()
    X_scaled= sc.fit_transform(X)
    return sc, X_scaled
def split(X,y):
    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test
def train_model(X_train, X_test, y_train, y_test, model='svc'):
    if model=='svc':
        svc=LinearSVC(dual=True, penalty='l2', loss='hinge')
        svc.fit(X_train, y_train)
        print('Test_score: ', svc.score(X_test,y_test))
        return svc
    elif model=='adaboost':
        ada=AdaBoostClassifier(n_estimators=5)
        ada.fit(X_train,y_train)
        print('Test_score: ', ada.score(X_test, y_test))
        return ada
    elif model =='xgboost':
        xgboost=xgb.XGBClassifier(objective='binary:logistic')
        xgboost.fit(X_train, y_train)
        y_hat= xgboost.predict(X_test)
        print(f'accuracy score: {accuracy_score(y_test, y_hat)}')
        return xgboost
    elif model== 'svc_nu':
        svc_nu=NuSVC(nu=0.25)
        svc_nu.fit(X_train, y_train)
        print(f'Test score: {svc_nu.score(X_test, y_test)}')
        return svc_nu
def save_model(file, svc,sc,params,y):
    os.chdir("/Users/datle/Desktop/license_plate_detection/train_vehicle_detection")
    with open(file, 'wb') as pfile:
        pickle.dump(
            {'svc': svc,
             'scaler': sc,
             'color_space': params['color_space'],
             'orient': params['orient'],
             'pix_per_cell': params['pix_per_cell'],
             'cell_per_block': params['cell_per_block'],
             'test_size': params['test_size'],
             # 'num_of_feature': svc.coef_.shape[-1],
             'size_of_pic_train': params['size_of_window'],
             'total_of_example': len(y),
             # 'model_parameter': svc.get_params()
             },
            pfile, pickle.HIGHEST_PROTOCOL)
    os.chdir("/Users/datle/Desktop/license_plate_detection")


### SLINDING WINDOW ###
def load_classifier(name, path):
    os.chdir(path)
    d= pickle.load(open(name, 'rb'))
    return d

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
                                         feature_vector=False, special=True, color_space=params['color_space'])
            feature_scale=scaler.transform(np.array(feature).reshape(1,-1))
            prediction= model.predict(feature_scale)
            if prediction==1:
                bboxs.append([int(x_pos*scale),int(y_pos*scale),int(scale*(x_pos+size_of_window[1])),int(scale*(y_pos*scale+size_of_window[0]))])
                # cv2.rectangle(img,(x_pos,y_pos),(x_pos_end,y_pos_end),(255,0,0))
                # cv2.imshow('r', img)
                # cv2.waitKey(1)
                # time.sleep(0.05)
    return bboxs
def find_car_multi_scale(img,params, win_size):
    bboxes=[]
    # print('number of scale use:', len(win_size['use_scale']))
    win_scale=win_size['use_scale']
    y_start_stop= win_size['y_start_stop']
    for x in range(win_size['length']):
        if x in win_scale:
            scale_0=win_size[f'scale_{x}'][2]
            bbox=sliding_window(img, params=params, y_start_stop= y_start_stop, overlap_rate=0.5, scale=scale_0)
            print(bbox)
            if len(bbox)==0:
                continue
            bboxes.append(bbox)
    if len(bboxes) ==0:
        return None,None
    bboxes= np.concatenate(bboxes)
    return bboxes
def draw(img,box):
    for x in box:
        cv2.rectangle(img, (x[0],x[1]), (x[2],x[3]), (0,0,255), 2)
    return img
def draw_heatmap(bbox,img):
    img_new= np.zeros_like(img)
    for box in bbox:
        img_new[box[1]:box[3],box[0]:box[2]]+=1
    print('threshhold: ',img_new[...,1].max())
    return img_new
def apply_threshhold(heatmap,thresh=3):
    heatmap=np.copy(heatmap)
    heatmap[heatmap< thresh]=0
    heatmap= np.clip(heatmap,0,254)
    return heatmap
def get_labeled(heatmap_thresh):
    labels= label(heatmap_thresh)
    bboxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ([np.min(nonzerox), np.min(nonzeroy), np.max(nonzerox), np.max(nonzeroy)])
        bboxes.append(bbox)
    # Return list of bounding boxes
    return bboxes

def product_heat_and_label_pic(heatmap, labels):
    # prepare RGB heatmap image from float32 heatmap channel
    img_heatmap = (np.copy(heatmap) / np.max(heatmap) * 255.).astype(np.uint8)
    img_heatmap = cv2.applyColorMap(img_heatmap, colormap=cv2.COLORMAP_JET)
    img_heatmap = cv2.cvtColor(img_heatmap, cv2.COLOR_BGR2RGB)

    # prepare RGB labels image from float32 labels channel
    img_labels = (np.copy(labels) / np.max(labels) * 255.).astype(np.uint8)
    img_labels = cv2.applyColorMap(img_labels, colormap=cv2.COLORMAP_HOT)
    img_labels = cv2.cvtColor(img_labels, cv2.COLOR_BGR2RGB)

    return img_labels, img_heatmap



### COMBINE SLIDING WINDOW FUNC ###
def filter_plate(bbox):
    for i,x in enumerate(bbox):
        # if (x[3]- x[1]) <15 and (x[2]- x[0]) <50:
        #     continue
        width= x[2]- x[0]
        height= x[3]- x[1]
        ratio= np.round_(width/height)
        if ratio in (1,):
            if (width >60) and (width<100) and (height >60) and (height <100):
                bbox[i]=x
    return bbox
def run(name,params, win_size, debug=False):
    if type(name)!=str:
        img=name
    else:
        img = cv2.imread(name, cv2.IMREAD_COLOR)
    img= cv2.resize(img, (1000,500))
    img2  = img.copy()
    start= time.time()
    bbox= find_car_multi_scale(img,params, win_size)
    if bbox is None:
        return img2, None
    end= time.time()
    print(f'time is: {end-start}')
    heatmap=draw_heatmap(bbox, img)
    heatmap_thresh= apply_threshhold(heatmap, thresh=win_size['thresh'])
    bbox_heatmap= get_labeled(heatmap_thresh)
    bbox_heatmap=filter_plate(bbox_heatmap)
    img2 = draw(img2, bbox_heatmap)
    if debug != False:
        heatmap_thresh, heatmap = product_heat_and_label_pic(heatmap, heatmap_thresh)
        ig=img.copy()
        img1= draw(ig, bbox)
        i= np.concatenate((img,img1,img2),axis=0)
        i1= np.concatenate((heatmap, heatmap_thresh), axis=0)
        i1= cv2.resize(i1, (600,300))
        cv2.imshow('i',i)
        cv2.imshow('i1',i1)
    return img2, bbox_heatmap
def test(params, win_size):
    os.chdir("/Users/datle/Desktop/Official_license_plate")
    l=glob.glob("./Training_vehicle_detection/result/run_load_data.jpeg")
    random.shuffle(l)
    for i in l:
        result,bbox= run(i,params,win_size,debug=True)
        cv2.imshow('r', result)
        cv2.waitKey(0)