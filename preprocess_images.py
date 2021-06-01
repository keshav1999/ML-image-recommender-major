import os
from PIL import Image , ImageOps
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
from keras.preprocessing import image
from keras.utils import np_utils

import numpy as np
import pandas as pd
import os
import cv2
from datetime import datetime
from datetime import date
import pickle



dirs = os.listdir('model_images')
folder_path = ""
image_data = []
labels = []
label_dict = {} 
dict_label = {}
for i in range(len(dirs)):
  label_dict[dirs[i]]=i
  dict_label[i] = dirs[i]

count = 0
co = 0
for ix in dirs:
    path = os.path.join('model_images',ix)
    img_data = os.listdir(path)
    count=count+1
    for im in img_data:
        img = image.load_img(os.path.join(path,im),target_size=(224,224))      
        img_array = image.img_to_array(ImageOps.grayscale(img))
        image_data.append(img_array)
        labels.append(label_dict[ix])
        co+=1
        print(count,co)

X_train = np.asarray(image_data)
Y_train = np.asarray(labels)
Y_train = np_utils.to_categorical(Y_train)

print(X_train.shape,Y_train.shape)
date = str(datetime.now())
path = os.path.join('TrainData',date +'.npy')
path2 = os.path.join('Label',date+'.npy')
path3 = os.path.join('LabelMap',date)
path4 = os.path.join('TrueLabel',date)

with open(path, 'wb') as f:
    np.save(f, X_train)
with open(path2, 'wb') as f:
    np.save(f, Y_train)
f = open(path3+".pkl","wb")
pickle.dump(dict_label,f)
f.close()
f = open(path4+".pkl","wb")
pickle.dump(labels,f)
f.close()
