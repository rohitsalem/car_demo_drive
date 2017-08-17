#!usr/bin/env python 
#Author: Rohit
#purpose : to verify the model with a sample image 

import numpy as np 
import tensorflow as tf 

import processData
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential 
from keras.layers import Dense 
import h5py
import cv2

np.random.seed()

data = pd.read_csv(processData.dataPath)
id = np.random.randint(0,len(data))

image = cv2.imread(processData.imPath+data.iloc[id]['center'].strip())
angle = data.iloc[id]['steering']
print ("actual angle: %f" %angle)

f = h5py.File('model.h5' , mode ='r')
model = load_model ('model_checkpoint.h5')
print (" ######### Loaded Model #########")

graph = []
graph.append(tf.get_default_graph())

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_array = np.asarray(image)

image_array = processData.crop (image_array, 0.3, 0.27)

image_array = processData.resize(image_array, new_dim=(64,64))

transformed_image_array = image_array[None, :, :, :]

steering_angle = float(model.predict(transformed_image_array, batch_size=1)) 

print ("actual angle: %f" %angle)

print ("predicted angle: %f" %steering_angle)