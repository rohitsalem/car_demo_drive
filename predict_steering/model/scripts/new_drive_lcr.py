#!/usr/bin/env python
## Author: Rohit
## Date: August 19, 2017
## Purpose: Drive node (inference)

import tensorflow as tf 
import rospy
import roslib
import cv2
import sys 
import argparse
import numpy as np 
import os,sys
from std_msgs.msg import Float32
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge, CvBridgeError

import json 
import h5py

from keras.models import load_model
from keras.models import model_from_json

import keras.backend.tensorflow_backend as KTF

GPU_FRACTION = 0.3

def get_session(gpu_fraction=GPU_FRACTION):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

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
			steering_angle = float (model.predict(image_array[None,:,:,:],batch_size=1))*-3

		self.pub.publish(steering_angle)





if __name__ == '__main__':
	
	MODEL_NAME = 'new_model_checkpoint_lcr.json'
	MODEL_PATH = os.path.join(os.path.dirname(sys.path[0]),'scripts', MODEL_NAME)
	print MODEL_PATH
	
	with open(MODEL_PATH, 'r') as jfile:
		model = model_from_json(json.loads(jfile.read()))


	model.compile("adam","mse")
	weights_file = MODEL_PATH.replace('json','h5')
	model.load_weights(weights_file)

	drive_obj = drive()

	rospy.init_node('drive_node')

	try:
		rospy.spin()
	except KeyboardInterrupt:
		print ("Shutdown")

	KTF.clear_session()
	cv2.destroyAllWindows()


