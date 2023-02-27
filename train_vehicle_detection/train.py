from ultils import *
from setting import params,win_size
name1=["/Users/datle/Desktop/ds/dataset_car/vehicles/GTI_Far/*.png"]
name1.append("/Users/datle/Desktop/ds/dataset_car/vehicles/GTI_Right/*.png")
name1.append("/Users/datle/Desktop/ds/dataset_car/vehicles/GTI_Left/*.png")
name1.append("/Users/datle/Desktop/ds/dataset_car/vehicles/GTI_MiddleClose/*.png")
name1.append("/Users/datle/Desktop/ds/dataset_car/vehicles/KITTI_extracted/*.png")

name2=["/Users/datle/Desktop/ds/dataset_car/non-vehicles/GTI_Far/*.png"]
name2.append("/Users/datle/Desktop/ds/dataset_car/non-vehicles/GTI_Left/*.png")
name2.append("/Users/datle/Desktop/ds/dataset_car/non-vehicles/GTI_MiddleClose/*.png")
name2.append("/Users/datle/Desktop/ds/dataset_car/non-vehicles/GTI_Right/*.png")

car, non_car= load_dataset(name1, name2, num_ex=10)
car_extreacted=extract_feature(car,params)
non_car_extracted= extract_feature(non_car, params)
X,y=combine(car_extreacted,non_car_extracted)

sc, X_scaled=normalize(X)

X_train, X_test, y_train, y_test= split(X_scaled, y)
print('start to train model')
model= train_model(X_train, X_test, y_train, y_test, model=params['model_name'])
save_model(file=f'ver1.p', svc= model,sc= sc, params=params,y=y)


