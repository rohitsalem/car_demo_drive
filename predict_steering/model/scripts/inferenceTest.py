#!usr/bin/env python 
#Author: Rohit
#purpose : to verify the model with a sample image 

import numpy as np 
import tensorflow as tf 

import processData
from keras.models import load_model

from keras.models import Sequential 
from keras.layers import Dense 
import h5py
import cv2

f = h5py.File('model.h5' , mode ='r')
model = load_model ('model.h5')
print (" ######### Loaded Model #########")

graph = []
graph.append(tf.get_default_graph())



image = cv2.imread('/home/rsalem/catkin_ws/src/priusCup/steering_predict/dataset/center/ocS2BqGJjH.png',-1)

# cv2.imshow('image',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

image_array = np.asarray(image)

image_array = processData.crop (image_array, 0.4, 0.2)

image_array = processData.resize(image_array, new_dim=(64,64))

transformed_image_array = image_array[None, :, :, :]

steering_angle = float(model.predict(transformed_image_array, batch_size=10)) 

print (steering_angle	)