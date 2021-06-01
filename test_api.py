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
import flask
from flask import render_template, request, jsonify,Flask
import io
import pickle
import time
import json   

matrix = np.load('features.npy')

with open("LabelMap/"+os.listdir("LabelMap")[0], 'rb') as f:
    labels = pickle.load(f)
with open("TrueLabel/"+os.listdir("TrueLabel")[0], 'rb') as f:
    true_labels = pickle.load(f)
def find20(vector):
    dist = []
    for j in range(matrix.shape[0]):
        dist.append((np.linalg.norm(matrix[j]-vector),j))
    dist.sort()
    return dist[0:20]
def modelmaking():
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
    return model_cut


my_model = modelmaking()


app = Flask(__name__,template_folder='templates')
 
@app.route('/', methods=['POST'])
def predict():

    if flask.request.method == 'POST':
        d1 = time.time()
        f = request.files["img"].read()
        f = Image.open(io.BytesIO(f))
        rgb_im = f.convert('RGB')
        rgb_im.save('audacious.jpg')
        img = f.resize((224,224))
        img = image.img_to_array(ImageOps.grayscale(img))
        print(img.shape)
        img = np.repeat(img[..., np.newaxis], 3, -1)
        img = img.reshape(1,224,224,3)
        print(img.shape)
        test_vector = my_model.predict(img).reshape(2500)
        print(test_vector.shape) 
        distances = find20(test_vector)
        print(distances)
        print(len(labels),type(labels))
        returnoutput = []
        for i in range(10):
            returnoutput.append((labels[true_labels[distances[i][1]]],distances[i][0]))
        Final_dict = {}
        for pi in returnoutput:
            if pi[0] not in Final_dict:
                Final_dict[pi[0]]= (pi[1],1)
            else:
                Final_dict[pi[0]]=(Final_dict[pi[0]][0]+pi[1],Final_dict[pi[0]][1]+1)
        for value in Final_dict:
            Final_dict[value] = int(Final_dict[value][0]//Final_dict[value][1])
        print(returnoutput)
        print(Final_dict)
        d2=time.time()
        print("Total time taken is ",str(d2-d1))
        initdict = {}
        for j in Final_dict:
        	varr = os.listdir("/home/keshav/Desktop/homecolor/model_images/"+j)
        	for k in varr:
        		name = j+"/"+k
        		initdict[name] = 0

        json_object = json.dumps(initdict) 
        with open("/home/keshav/Desktop/nn/Image-Recommendor-master/src/data.json", 'w') as fp:
        	json.dump(initdict, fp)
          
        print(json_object)
        return json_object

if __name__ == "__main__":
   app.run()
        
