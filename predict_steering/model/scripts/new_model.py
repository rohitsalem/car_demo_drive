#!/usr/bin/python

import tensorflow as tf 
import numpy as np 
import processimage
import json
import h5py

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dense, Dropout, ELU, Flatten, Input, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.models import Sequential, Model, load_model, model_from_json
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint 

dataPath =  "../../dataset/yaml_files/data_new.csv"

def get_model():
	model = Sequential([
	# Normalize image to -1.0 to 1.0
		Lambda(processimage.normalize_image, input_shape=(66, 200, 3)),
		# Convolutional layer 1 24@31x98 | 5x5 kernel | 2x2 stride | elu activation 
		Convolution2D(24, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)),
		# Dropout with drop probability of .1 (keep probability of .9)
		Dropout(.1),
		# Convolutional layer 2 36@14x47 | 5x5 kernel | 2x2 stride | elu activation
		Convolution2D(36, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)),
		# Dropout with drop probability of .2 (keep probability of .8)
		Dropout(.2),
		# Convolutional layer 3 48@5x22  | 5x5 kernel | 2x2 stride | elu activation
		Convolution2D(48, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)),
		# Dropout with drop probability of .2 (keep probability of .8)
		Dropout(.2),
		# Convolutional layer 4 64@3x20  | 3x3 kernel | 1x1 stride | elu activation
		Convolution2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1), init='he_normal', W_regularizer=l2(0.001)),
		# Dropout with drop probability of .2 (keep probability of .8)
		Dropout(.2),
		# Convolutional layer 5 64@1x18  | 3x3 kernel | 1x1 stride | elu activation
		Convolution2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1), init='he_normal', W_regularizer=l2(0.001)),
		# Flatten
		Flatten(),
		# Dropout with drop probability of .3 (keep probability of .7)
		Dropout(.3),
		# Fully-connected layer 1 | 100 neurons | elu activation
		Dense(100, activation='elu', init='he_normal', W_regularizer=l2(0.001)),
		# Dropout with drop probability of .5
		Dropout(.5),
		# Fully-connected layer 2 | 50 neurons | elu activation
		Dense(50, activation='elu', init='he_normal', W_regularizer=l2(0.001)),
		# Dropout with drop probability of .5
		Dropout(.5),
		# Fully-connected layer 3 | 10 neurons | elu activation
		Dense(10, activation='elu', init='he_normal', W_regularizer=l2(0.001)),
		# Dropout with drop probability of .5
		Dropout(.5),
		# Output
		Dense(1, activation='linear', init='he_normal')
		])

	model.compile(optimizer=Adam(0.00001), loss='mse')
	model.load_weights('new_model2.h5')
	return model



if __name__=="__main__":

	X_train,y_train = processimage.get_csv_data(dataPath)
	X_train, y_train = shuffle(X_train,y_train, random_state=14)
	X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.3, random_state=14)

	model = get_model()
	model.summary()
	filepath = "new_model_checkpoint.h5"
	checkpoint = ModelCheckpoint(filepath, monitor = "loss", verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	model.fit_generator(processimage.generate_batch(X_train, y_train), steps_per_epoch=200, 
    					epochs=20,
    					verbose=1,
    					validation_data=processimage.generate_batch(X_validation, y_validation), 
    					validation_steps=20,
    					callbacks = callbacks_list)

	print ('Saving model weights and configuration file')

	model.save_weights('new_model.h5')
    # Save model architecture as json file
	with open('new_model.json', 'w') as outfile:
		json.dump(model.to_json(), outfile)

	with open('new_model_checkpoint.json', 'w') as outfile:
		json.dump(model.to_json(), outfile)
    # Explicitly end tensorflow session
	from keras import backend as K 

	K.clear_session()	