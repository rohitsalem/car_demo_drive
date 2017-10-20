#!/usr/bin/python
##Author: Rohit
##Purpose: process images for center left and right images : in that order

import os
import cv2
import random
import numpy as np
import pandas as pd

def get_my_path():
    try:
        filename = __file__ # where we were when the module was loaded
    except NameError: # fallback
        filename = inspect.getsourcefile(get_my_path)
    return os.path.realpath(filename)

# path to this script
cm_path = get_my_path()
# go 3 directory levels up
sp_path = reduce(lambda x, f: f(x), [os.path.dirname]*3, cm_path)
dataPath = os.path.join(sp_path, "dataset","yaml_files", "data_lcr.csv")
imPath = os.path.join(sp_path, "dataset")

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

	data = pd.read_csv(file)
	image_names, steering_angles= [],[]
	steering_offest = 0.27        # for left and right steering corrections
	image_names = [list(data['center']), list(data['left']), list(data['right'])]
	center_angles = list(data['angle'])
	left_angles = []
	for i in center_angles:
		left_angles +=[i-0.27]
 	right_angles = []
 	for i in center_angles:
 		right_angles +=[i+0.27]
	steering_angles = [center_angles,left_angles,right_angles]

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
		image = X_train[index]

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
			ind  = np.random.randint(0,3)
			raw_image = cv2.imread(imPath+image_file[ind])
			raw_angle = float(angle[ind])
			image = process_image(raw_image)
			# to randomly
			# if random.randrange(4)==1:
			# 	image = cv2.flip(image,1)
			# 	raw_angle = -raw_angle
			X_batch.append(image)
			y_batch.append(raw_angle)


		assert len(X_batch) == batch_size, 'len(X_batch) == batch_size should be True'

		yield np.array(X_batch), np.array(y_batch)



# def show_processedimages():
# 	images,angles = get_csv_data(dataPath)
# 	id = np.random.randint(0,len(images))
# 	print ("id: %d %f" %(id,angles[id][0]) )

# 	img = cv2.imread(imPath +str(images[id][0]))
# 	img=process_image(img)
# 	cv2.imshow("image" , img)
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()

# if __name__=="__main__":
# 	show_processedimages()
