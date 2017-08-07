import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocess import data_iterator
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Lambda, Convolution2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.optimizers import Adam

import os

# Prepare datasets
input_shape = (64, 64, 3)
data_dir = "/home/rohit/indigo_ws/src/car/predict_steering/dataset/center"
csv_dir= "/home/rohit/indigo_ws/src/car/predict_steering/dataset/yaml_files"
data_df = pd.read_csv(os.path.join(csv_dir, 'data.csv'))
X = data_df['center'].values
y = data_df['steering'].values
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10)

# Define convnet
model = Sequential()
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=input_shape))
model.add(Convolution2D(24, (5, 5), activation='relu', strides=(2, 2)))
model.add(Convolution2D(36, (5, 5), activation='relu', strides=(2, 2)))
model.add(Convolution2D(48, (5, 5), activation='relu', strides=(2, 2)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()

model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-4))
history = model.fit_generator(generator=data_iterator(data_dir, X_train, y_train, 64, True),
                    steps_per_epoch=800,
                    epochs=10,
                    max_q_size=10,
                    validation_data=data_iterator(data_dir, X_valid, y_valid, 64, False),
                    validation_steps=50,
                    verbose=1)
model.save("model.h5")
