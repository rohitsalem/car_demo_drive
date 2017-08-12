#!/usr/bin/python
## Author: rohit
## Date: June, 29,2017
## Model for training

from __future__ import print_function
from keras.preprocessing.image import *
from keras.models import Sequential ,Model, load_model
from keras.layers import Dense, Lambda, Dropout, Activation, Flatten, SpatialDropout2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
from keras import initializers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np 
import json
import math

import processData

number_of_epochs =40
number_of_steps_per_epoch =2500
number_of_validation_steps = 500
learning_rate = 1e-4

activation_relu = 'relu'

# model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
model = Sequential()

#change the input_shape accordingly
model.add(Lambda(lambda x: x / 127.5 - 1.0,  input_shape=(64, 64, 3)))

# starts with five convolutional and maxpooling layers
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())

# Next, five fully connected layers
model.add(Dense(1164))
model.add(Activation(activation_relu))

model.add(Dense(100))
model.add(Activation(activation_relu))

model.add(Dense(50))
model.add(Activation(activation_relu))

model.add(Dense(10))
model.add(Activation(activation_relu))

model.add(Dense(1))

model.summary()

model.compile(optimizer=Adam(learning_rate), loss="mse")

#Saving the model architecture
#model_json = model.to_json()
#with open("model.json", "w") as outfile:
#	outfile.write(model_json)

# Load the pre-trained weights
model.load_weights('weights.h5')
#model=load_model('model.h5')
print ("Loaded the pre trained weights")


# create two generators for training and validation
trainGen = processData.genBatch()
valGen = processData.genBatch()
evalGen = processData.genBatch()

filepath = "model_checkpoint.h5"
checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
history = model.fit_generator(trainGen,
                              steps_per_epoch=number_of_steps_per_epoch,
                              epochs=number_of_epochs,
                              verbose=1,
                              validation_data=valGen,
                              callbacks=callbacks_list,
                             validation_steps=number_of_validation_steps)
# #
score = model.evaluate_generator(evalGen, 2, max_queue_size=10)
print('Test score:', score)
# # print('Test accuracy:', score[1])

# # save the weights
model.save_weights('weights.h5')

# #save the model with weights
model.save('model.h5')

# test_img= '../../dataset/center/QvnkdWI64B.png'
# def visualize_layer(layer_name):
# 	modeli=Model(input=model.input, output=model.get_layer(layer_name).output)

# 	img= load_img(test_img)
# 	# img=processData.crop(img,0.3,0.27)
# 	# img=processData.resize(img,(64,64))
# 	# im=np.asarray(img)
# 	img=processData.crop_resize_image(img_to_array(img))
# 	# img=np.asarray(img)
# 	img=np.expand_dims(img,axis=0)

# 	print(img.shape)

# 	conv_features=modeli.predict(img)
# 	print("conv features shape:", conv_features.shape)


# 	plt.subplots(figsize=(5,5))
# 	for i in range(16):
# 		plt.subplot(4,4,i+1)
# 		plt.axis('off')
# 		plt.imshow(conv_features[0,:,:,i],cmap='gray')
# 	plt.show()

# visualize_layer('conv2d_3')
