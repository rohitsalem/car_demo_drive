# Demo of Autonomous driving of Prius in ROS/GAZEBO

This is a simulation of a Prius in [gazebo 8](http://gazebosim.org) with sensor data being published using [ROS kinetic](http://wiki.ros.org/kinetic/Installation)
The car's throttle, brake, steering, and gear shifting are controlled by publishing a ROS message.
A ROS node allows driving with a gamepad or joystick.

# Requirements

This demo has been tested on Ubuntu Xenial (16.04)

* An X server
* [Docker](https://www.docker.com/get-docker)
* [nvidia-docker](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-1.0))

# Running directly (inference)
To have the car drive autonomously using the pre-trained weights, \
Make sure the cruize_control flag is set to `true` [here](https://github.com/rohitsalem/car_demo_drive/blob/master/prius_description/urdf/prius.urdf#L453). This allows the car to drive at a constant velocity.\
RUN:  `roslaunch car_demo demo_drive.launch` This will load the Sonoma Raceway track and along with prius. Here the car will drive around the track by subscribing to `/steering_command` from the model prediction.   

# Steps for training and testing new models: 
## Using only the front facing Center camera for training:
**Recording Bag Files and data generation**
- Make sure the cruize_control flag is set to `false` [here](https://github.com/rohitsalem/car_demo_drive/blob/master/prius_description/urdf/prius.urdf#L453).
- RUN `roslaunch car_demo demo_record_center.launch` 
- Open a terminal and navigate to `car/predict_steering/dataset`, Here we store all the data required for training, (i.e images and the csv files) and RUN: 
```
rosbag record /filtered/steering_angle /filtered/image_raw 
```
This will save the rosbag file with images from the center front facing camera on the car along with the corresponding steering angle. 
- Start driving the car using `WASD` or a joystick. End the recording of rosbag file when you have traversed enough in the track. Record more rosbag files (traversing the same track 4-5 times), this will allow enough data for training. 
- Once we have the bag files, we have to extract them into images and steering angles. RUN `rosrun dataprocess bag_extract_data_center.py`, this will generate a csv file with path to the images from center camera along with the corresponding steering angles.  

**Training the Model** 
- To train the model, RUN `rosrun model train_center.py` , this will start training the model using the data generated from the previous process. 
- After each epoch, only the best weights are saved automatically, i.e weights are saved only if there is an improvement in training accuracy from the previous epochs.

**Inference (testing on the car)**
- RUN `roslaunch car_demo demo_drive.launch` to test the model on the car. 

## Using the three front facing cameras of the car: Center, left and right for training:
**Recording Bag Files and data generation**
- RUN `roslaunch car_demo demo_record_lcr.launch` 
- Open a terminal and navigate to `car/predict_steering/dataset`, Here we store all the data required for training, (i.e images and the csv files) and RUN: 
```
rosbag record /filtered/steering_angle /filtered/center/image_raw /filtered/left/image_raw /filtered/right/image_raw 
```
This will save the rosbag file with images from three front facing cameras on the car along with the corresponding steering angle. 
- Start driving the car using `WASD` or a joystick. End the recording of rosbag file when you have traversed enough in the track. Record more rosbag files (traversing the same track 4-5 times), this will allow enough data for training. 
- Once we have the bag files, we have to extract them into images and steering angles. RUN `rosrun dataprocess bag_extract_data_lcr.py`, this will generate a csv file with path to the images from all three cameras along with the corresponding steering angles.  

**Training the Model** 
- To train the model, RUN `rosrun model train_lcr.py` , this will start training the model using the data generated from the previous process. 
- After each epoch, only the best weights are saved automatically, i.e weights are saved only if there is an improvement in training accuracy from the previous epochs.

**Inference (testing on the car)**
- RUN `roslaunch car_demo demo_drive.launch` to test the model on the car. 





 
