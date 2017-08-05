#!/usr/bin/python
## Author: sumanth
## Date: Nov, 20,2016
## Purpose: reads the data set bag file, generates the images into left, center, right
## and generates a text file with the corresponding steering and velocity values

# This code has been derived from (https://github.com/DominicBreuker/self-driving-car-experiments#getting-data) 

import os
import rosbag
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rospkg

#set rospack
rospack = rospkg.RosPack()
#get package
data_dir=rospack.get_path('dataset')
rosbag_file = os.path.join(data_dir, "dataset.bag")

def get_image_dir(image_type):
    images_dir = os.path.join(data_dir, image_type)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    return images_dir

#left_images_dir = get_image_dir("left")
center_images_dir = get_image_dir("center")
#right_images_dir = get_image_dir("right")


steering_report_topic="/gazebo_client/steering_angle"
center_camera_topic="/prius/camera1/image_raw"
#steering_report_topic = "/vehicle/steering_report"
#left_camera_topic = "/left_camera/image_color"
#center_camera_topic = "/center_camera/image_color"
#right_camera_topic = "/right_camera/image_color"
topics = [steering_report_topic, center_camera_topic]
#          center_camera_topic, right_camera_topic]

center_images_dir=get_image_dir("center")


angle_timestamps = []
angle_values = []
speed_values = []
image_names = []
images = []
tmp = []

image_timestamps = []

bridge = CvBridge()

def save_image(dir, msg, image_name):
    path_name = os.path.join(dir, image_name)
    try:
        cv2.imwrite(path_name, bridge.imgmsg_to_cv2(msg))
    except CvBridgeError as e:
        print(e)

#count used for image numbering
im_count = 1
#Use this variable to filter images which has not much steering movement.
angle_threshold = 0 #0.523599 # This value is equal to 30 degree. Set this value to zero, if you are want to save all the images
#Use average of angles until image timestamp
use_average = True

with rosbag.Bag(rosbag_file, "r") as bag:
    for topic, msg, t in bag.read_messages(topics=topics):
        if topic == steering_report_topic:
            angle_values.append(msg.data) 
            print('steering wheel angle 1')
            print (msg.data)
        elif topic == center_camera_topic:
            image_name = '{}{:08d}{}'.format("image_", im_count, ".png")
            save_image(center_images_dir, msg, image_name)
            image_names.append(image_name)
        # tmp.append(angle_value)
            # if 0 in speed_values:
            #     if use_average:
            #         angle_value = np.mean(angle_values)
            #         if abs(angle_value) > angle_threshold:
            #             #print ""
            #             image_names.append(image_name)
            #             tmp.append(angle_value)
            #             #save this image
            #     else:
            #         # at any image timestamp, use the last known steering angle
            #         angle_value = angle_values[-1]
            #         if abs(angle_value) > angle_threshold:
            #             print ""
            #             image_names.append(image_name)
            #             tmp.append(angle_value)
            #             # save this image

            # reset the list to empty. Cant think of a better way to do it
        # angle_values[:] = []
        # speed_values[:] = []
        images[:] = []
        im_count = im_count + 1
        if im_count % 1000 == 0:
            print 'Done processing images :{}'.format(im_count)

steering_angle_file = os.path.join(data_dir, "image_steering_angle.txt")

with open(steering_angle_file, "w") as data_file:
    for idx in range(len(angle_values)):
        line = '{},{}\n'.format(image_names[idx], angle_values[idx])
        data_file.write(line)
    data_file.close()
