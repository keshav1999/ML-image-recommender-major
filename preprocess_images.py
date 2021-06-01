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
        img1 = img.rotate(90)
        img2 = img.rotate(180)
        img3 = img.rotate(270)
        img4 = ImageOps.mirror(img)
        img5 = img4.rotate(90)
        img6 = img4.rotate(180)
        img7 = img4.rotate(270)
              
        img_array = image.img_to_array(ImageOps.grayscale(img))
        img_array1 = image.img_to_array(ImageOps.grayscale(img1))
        img_array2 = image.img_to_array(ImageOps.grayscale(img2))
        img_array3 = image.img_to_array(ImageOps.grayscale(img3))
        img_array4 = image.img_to_array(ImageOps.grayscale(img4))
        img_array5 = image.img_to_array(ImageOps.grayscale(img5))
        img_array6 = image.img_to_array(ImageOps.grayscale(img6))
        img_array7 = image.img_to_array(ImageOps.grayscale(img7))
        image_data.append(img_array)
        image_data.append(img_array1)
        image_data.append(img_array2)
        image_data.append(img_array3)
        image_data.append(img_array4)
        image_data.append(img_array5)
        image_data.append(img_array6)
        image_data.append(img_array7)
        for i in range(8):
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
