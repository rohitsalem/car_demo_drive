#!usr/bin/env python

#Author: Rohit
#Date: June, 30, 2017
#Purpose: Inference code for publishing steering angle
from __future__ import print_function
#import roslib
#roslib.load_manifest('model')
import rospy
import sys
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf


import argparse
import json

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.models import load_model
import h5py
from keras import __version__ as keras_version
import processData
from keras.models import Sequential
from keras.layers import Dense

model=None
prev_image_array=None



class drive:

	def __init__(self):
		self.pub = rospy.Publisher("steering_command", Float32, queue_size = 10)

		self.bridge = CvBridge()

		self.sub = rospy.Subscriber("/prius/camera/center/image_raw", Image, self.callback)

		# self.json_file = open('model.json', 'r')
		
		# self.loaded_model_json = self.json_file.read()
		# self.json_file.close()
		# self.model = model_from_json(self.loaded_model_json)
		
		# self.model.load_weights("weights.h5")
		# print("Loaded weights from disk")
		
		# self.model.compile(loss='mse', optimizer='adam')
		# print ("model compiled")

		self.f = h5py.File('model.h5' , mode ='r')
		self.model = load_model ('model.h5')
		print (" ######### Loaded Model #########")

		self.graph = []
		self.graph.append(tf.get_default_graph())


	def callback(self,data):

		try:
			cv_image=self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
		except CvBridgeError as e:
			print(e)

		# cv2.imshow("Image window", cv_image)
		# cv2.waitKey(3)

		image_array = np.asarray(cv_image)

		image_array = processData.crop (image_array, 0.35, 0.1)

		image_array = processData.resize(image_array, new_dim=(64,64))

		transformed_image_array = image_array[None, :, :, :]

		with self.graph[0].as_default():
			steering_angle = float(self.model.predict(transformed_image_array, batch_size=1))
			self.pub.publish(steering_angle)


def main(args):
	obj = drive()
	rospy.init_node('drive_node')
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print ("Shutdown")
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main(sys.argv)
