#!usr/bin/env python
#Author: Rohit
import tensorflow as tf 
import rospy
import cv2
import sys 
import argparse
import numpy as np 

from std_msgs.msg import Float32
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge, CvBridgeError

import json 
import h5py

from keras.models import load_model
from keras.models import model_from_json



class drive:

	def __init__(self):
		self.pub = rospy.Publisher ("steering_command", Float32, queue_size = 1)
		self.bridge = CvBridge()
		self.sub = rospy.Subscriber ("/prius/front_camera/image_raw", Image, self.callback)
		self.graph = []
		self.graph.append(tf.get_default_graph())	

	def callback (self,data):

		try:
			cv_image =self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
		except CvBridgeError as e:
			print(e)

		image_array = np.asarray(cv_image)

		image_array = image_array[140:-120,:]
		image_array = cv2.resize(image_array, (200,66), interpolation = cv2.INTER_AREA)
		with self.graph[0].as_default():
			steering_angle = float (model.predict(image_array[None,:,:,:],batch_size=1))

		self.pub.publish(steering_angle)





if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description="Remote Driving")
	parser.add_argument(
		'model' ,
		type=str ,
		help = 'Path to model h5 file and Model should be on the same path')
	args = parser.parse_args()

	with open(args.model, 'r') as jfile:
		model = model_from_json(json.loads(jfile.read()))

	model.compile("adam","mse")
	weights_file = args.model.replace('json','h5')
	model.load_weights(weights_file)

	drive_obj = drive()

	rospy.init_node('drive_node')

	try:
		rospy.spin()
	except KeyboardInterrupt:
		print ("Shutdown")
	cv2.destroyAllWindows()


