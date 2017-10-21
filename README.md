# Demo of Autonomous driving of Prius in ROS/GAZEBO

This is a simulation of a Prius in [gazebo 8](http://gazebosim.org) with sensor data being published using [ROS kinetic](http://wiki.ros.org/kinetic/Installation)
The car's throttle, brake, steering, and gear shifting are controlled by publishing a ROS message.
A ROS node allows driving with a gamepad or joystick.

# Video + Pictures

A video and screenshots of the demo can be seen in this blog post: https://www.osrfoundation.org/simulated-car-demo/

![Prius Image](https://www.osrfoundation.org/wordpress2/wp-content/uploads/2017/06/prius_roundabout_exit.png)

# Requirements

This demo has been tested on Ubuntu Xenial (16.04)

* An X server
* [Docker](https://www.docker.com/get-docker)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/wiki/Installation)

# Running directly (inference)
To have the car drive autonomously using the pre-trained weights, \
RUN:  `roslaunch car_demo demo_drive.launch` This will load the Sonoma Raceway track and along with prius. Here the car will drive around the track by subscribing to `/steering_command` from the model prediction.   

# Steps for training and testing new models: 
## Using the front three cameras of the car: Center, left and right for data collection:
**Recording Bag Files**
- RUN `roslaunch car_demo demo_record.launch` 
- Open a terminal and navigate to `car/predict_steering/dataset`, Here we store all the data required for training, (i.e images and the csv files) and RUN: 
```
rosbag record /filtered/steering_angle /filtered/center/image_raw /filtered/left/image_raw /filtered/right/image_raw 
```
- Start driving the car using `WASD` or a joystick. 

 
