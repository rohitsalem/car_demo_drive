#!usr/bin/env python 
#Author: Rohit
#Date: July 18
#Purpose : publisher for verifying the data for 10 kmph (reading from data.csv and publishing to the steering_command)

from __future__ import print_function
import roslib
import rospy
import sys
import os 

from std_msgs.msg import Float32
import csv

with open('data.csv') as csvfile:
	datafile = csv.reader(csvfile, delimiter = ',')
	steering = []
	for col in datafile:
		steering.append( col[1])
#print (float(steering[1])*float(steering[2]))
print ("loaded the csv file")

def talker():

	pub = rospy.Publisher("steering_command", Float32, queue_size = 1)
	rospy.init_node('talker', anonymous=True)
	rate= rospy.Rate(59) 

	print("Inside Talker")
	for index in range(len(steering)):
		steering_angle=float(steering[index])
		pub.publish(steering_angle)
		rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass