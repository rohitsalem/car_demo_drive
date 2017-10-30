# Demo of Autonomous driving of Prius in ROS/GAZEBO

This is a simulation of a Prius in [gazebo 8](http://gazebosim.org) with sensor data being published using [ROS kinetic](http://wiki.ros.org/kinetic/Installation)
The car's throttle, brake, steering, and gear shifting are controlled by publishing a ROS message.
A ROS node allows driving with a gamepad or joystick.

# Requirements

This demo has been tested on Ubuntu Xenial (16.04)

* An X server
* [Docker](https://www.docker.com/get-docker)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/wiki/Installation)

# Running directly (inference)
To have the car drive autonomously using the pre-trained weights, \
RUN:  `roslaunch car_demo demo_drive.launch` This will load the Sonoma Raceway track and along with prius. Here the car will drive around the track by subscribing to `/steering_command` from the model prediction.   

# Steps for training and testing new models: 
## Using the front three cameras of the car: Center, left and right for training:
**Recording Bag Files and data generation**
- RUN `roslaunch car_demo demo_record.launch` 
- Open a terminal and navigate to `car/predict_steering/dataset`, Here we store all the data required for training, (i.e images and the csv files) and RUN: 
```
rosbag record /filtered/steering_angle /filtered/center/image_raw /filtered/left/image_raw /filtered/right/image_raw 
```
This will save the rosbag file with images from three front facing cameras on the car along with the corresponding steering angle. 
- Start driving the car using `WASD` or a joystick. End the recording of rosbag file when you have traversed enough in the track. Record more rosbag files (traversing the same track 4-5 times), this will allow enough data for training. 
- Once we have the bag files, we have to extract them into images and steering angles. RUN `rosrun dataprocess bag_extract_data_lcr.py`, this will generate a csv file with path to the images from all three cameras along with the corresponding steering angles.  

**Training the Model** 
- To train the model, RUN `rosrun model mode_lcr.py` , this will start training the model using the data generated from the previous process. 
- After each epoch, only the best weights are saved automatically, i.e weights are saved only if there is an improvement in training accuracy from the previous epochs.

**Inference (testing on the car)**
- RUN `roslaunch car_demo demo_drive.launch` to test the model on the car. 



 
