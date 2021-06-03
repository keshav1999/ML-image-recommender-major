import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.layers import Dense,Activation,Flatten,Input,Dropout,GlobalAveragePooling2D
from keras.models import Model
from keras.utils import np_utils
from keras.models import Model
import random
import numpy as np
import pandas as pd
import urllib
import os
from urllib.request import urlretrieve
import cv2
from datetime import datetime
from PIL import Image , ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import display


X_train = np.load("TrainData/"+os.listdir('TrainData')[0])

X_train = X_train.reshape(2296,224,224)

X_train = np.repeat(X_train[..., np.newaxis], 3, -1)
X_train = X_train.reshape(2296,224,224,3)
print(X_train.shape)
model = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3))
av1 = GlobalAveragePooling2D()(model.output)
fc1 = Dense(10024,activation='relu')(av1)
fc2 = Dense(5120,activation='relu')(fc1)
fc3 = Dense(2500,activation='relu')(fc2)
d1 = Dropout(0.5)(fc3)
fc4 = Dense(1257,activation='softmax')(d1)
model_new = Model(inputs=model.input, outputs= fc4)
adam = Adam(lr=0.0000007)
model_new.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model_new.load_weights('model15.h5')
model_cut = Model(inputs=model_new.inputs, outputs=model_new.layers[-3].output)
features = model_cut.predict(X_train)
print(features.shape)
with open("features.npy", 'wb') as f:
    np.save(f, features)