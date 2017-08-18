#!/usr/bin/python
##Author: Rohit
##Purpose: process images

import cv2
import numpy as np 
import random
import os
import csv
from scipy.stats import bernoulli


imPath = '../../dataset/center/'
BatchSize=64
def resize_image(image):

	return cv2.resize(image, (200,66), interpolation=cv2.INTER_AREA)


def normalize_image(image):
	
	return image/127.5 -1 


def crop_image (image):

	return image[140:-120,:]



def process_image(image):

	image = crop_image(image)
	image = resize_image(image)
	return image


def get_csv_data(file):

	image_names, steering_angles= [],[]

	with open(file, 'r') as f:
		reader = csv.reader(f)
		next(reader, None)
		for center_img, steering in reader:
			angle = float(steering)
			image_names.append(center_img.strip())
			steering_angles.append(angle)
	return image_names, steering_angles




def fetch_images(X_train, y_train, batch_size):

	thresh_prob=3
	thresh = 0.01
	count = 0
	zeros_count= 0
	images_and_angles=[]
	

	while (count < batch_size):

		index = np.random.randint(0,len(X_train))
		angle = y_train[index]
		image = str(X_train[index])

		if (-thresh < angle < thresh):
			if(zeros_count<15):
				images_and_angles.append((image,angle))
				zeros_count =zeros_count + 1
				count = count + 1

		else:
			images_and_angles.append((image,angle))
			count = count + 1

	return images_and_angles

def generate_batch(X_train, y_train, batch_size=BatchSize):

	while True:
		X_batch = []
		y_batch = []
		images_and_angles=fetch_images(X_train,y_train,batch_size)
		for image_file , angle in images_and_angles:
			raw_image = cv2.imread(imPath+image_file)
			raw_angle = float(angle)
			image = process_image(raw_image)
			# if random.randrange(2)==1:
			# 	image = cv2.flip(image,1)
			# 	raw_angle = -raw_angle
			X_batch.append(image)
			y_batch.append(raw_angle)


		assert len(X_batch) == batch_size, 'len(X_batch) == batch_size should be True'

		yield np.array(X_batch), np.array(y_batch)

	

# def show_processedimages():
# 	images,angles = get_csv_data(dataPath)
# 	id = np.random.randint(0,len(images))
# 	print ("id: %d %f" %(id,angles[id]) )
	
# 	img = cv2.imread(imPath +str(images[id]))
# 	img=process_image(img)
# 	cv2.imshow("image" , img)
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()

	
# def generate_batch(X_train, y_train, batch_size=10):
# 	print "generate batch"
# 	images = np.zeros((batch_size,66,200,3), dtype=np.float32)
# 	angles = np.zeros((batch_size,), dtype=np.float32)
# 	while True:
# 		straight_count = 0
# 		for i in range(batch_size):
# 			image_index = np.random.randint(0,len(X_train))
# 			angle = y_train[image_index]
# 			# Limit angles of less than absolute value of .1 to no more than 1/2 of data
# 			# to reduce bias of car driving straight
# 			if abs(angle) < .01:
# 			    straight_count += 1
# 			if straight_count > (batch_size * .5):
# 			    while abs(y_train[image_index]) < .1:
# 			        sample_index = random.randrange(len(X_train))
# 			# Read image in from directory, process, and convert to numpy array
# 			image = cv2.imread(imPath + str(X_train[image_index]))
# 			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 			image = process_image(image)
# 			image = np.array(image, dtype=np.float32)
# 			# Flip image and apply opposite angle 50% of the time
# 			if random.randrange(2) == 1:
# 			    image = cv2.flip(image, 1)
# 			    angle = -angle
# 			images[i] = image
# 			angles[i] = angle
# 			print  i 
# 	yield images, angles