#!/usr/bin/python
## Purpose: reads the data set bag file, generates the data

# code is derived from https://github.com/rwightman/udacity-driving-reader/tree/master/script

from __future__ import print_function
from cv_bridge import CvBridge, CvBridgeError
from collections import defaultdict
import os
import sys
import cv2
import imghdr
import argparse
import functools
import numpy as np
import pandas as pd
import rospkg
from bagutils import *
import random
import string
import csv

def get_outdir(base_dir, name):
    outdir = os.path.join(base_dir, name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

def write_image(bridge, outdir, msg, fmt='png'):
    results = {}
    random_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in xrange(10)])
    image_filename = os.path.join(outdir, str(random_name) + '.' + fmt) # File name is a random string
    # image_filename = os.path.join(outdir, str(msg.header.stamp.to_nsec()) + '.' + fmt)           # Filename is the timestamp
    try:
        if hasattr(msg, 'format') and 'compressed' in msg.format:
            buf = np.ndarray(shape=(1, len(msg.data)), dtype=np.uint8, buffer=msg.data)
            cv_image = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
            if cv_image.shape[2] != 3:
                print("Invalid image %s" % image_filename)
                return results
            results['height'] = cv_image.shape[0]
            results['width'] = cv_image.shape[1]
            cv2.imwrite(image_filename, cv_image)
        else:
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imwrite(image_filename, cv_image)
    except CvBridgeError as e:
        print(e)
    results['filename'] = image_filename
    return results


def camera2dict(msg, write_results, camera_dict, camera_name):
    if camera_name == "center":
        camera_dict["timestamp"].append(msg.header.stamp.to_nsec())
    # camera_dict["width"].append(write_results['width'] if 'width' in write_results else msg.width)
    # camera_dict['height'].append(write_results['height'] if 'height' in write_results else msg.height)
    # camera_dict["frame_id"].append(msg.header.frame_id)
    camera_dict[camera_name].append(write_results['filename'])


def steering2dict(msg, steering_dict):
    steering_dict["timestamp"].append(msg.header.stamp.to_nsec())
    steering_dict["angle"].append(msg.data)

def camera_select(topic, select_from):
	if topic.startswith('/filtered/l'):
		return select_from[0]
	if topic.startswith('/filtered/c'):
		return select_from[1]
	if topic.startswith('/filtered/r'):
		return select_from[2]
	else:
		assert False, "Unexpected Topic"


def main():
    #set rospack
    rospack = rospkg.RosPack()
    #get package
    data_dir=rospack.get_path('dataset')
    rosbag_file = os.path.join(data_dir, "dataset.bag")

    bridge = CvBridge()

    include_images = True
    include_others = False
    debug_print = False
    gen_interpolated = False
    img_format = 'png'

    LEFT_IMAGE_TOPIC = "/filtered/left/image_raw"
    CENTER_IMAGE_TOPIC = "/filtered/center/image_raw"
    RIGHT_IMAGE_TOPIC = "/filtered/right/image_raw"
    STEERING_TOPIC="/filtered/steering_angle"

    filter_topics = [STEERING_TOPIC ,  LEFT_IMAGE_TOPIC , CENTER_IMAGE_TOPIC , RIGHT_IMAGE_TOPIC]

    bagsets = find_bagsets(data_dir, filter_topics=filter_topics)
    for bs in bagsets:
        print("Processing set %s" % bs.name)
        sys.stdout.flush()

        dataset_outdir = os.path.join(data_dir, "%s" % bs.name)

        left_outdir = get_outdir(data_dir,"left")
        center_outdir = get_outdir(data_dir, "center")
        right_outdir = get_outdir(data_dir, "right")
        yaml_outdir = get_outdir(data_dir, "yaml_files")

        camera_dict_left = defaultdict(list)
        camera_dict_center = defaultdict(list)
        camera_dict_right = defaultdict(list)

        steering_cols = ["timestamp", "angle"]
        steering_dict = defaultdict(list)

        bs.write_infos(yaml_outdir)
        readers = bs.get_readers()
        stats_acc = defaultdict(int)

        def _process_msg(topic, msg, stats):
            timestamp = msg.header.stamp.to_nsec()
            if  (topic == LEFT_IMAGE_TOPIC) :
                outdir = left_outdir
                if debug_print:
                    print("%s_camera %d" % (topic[1],timestamp))
                results = write_image(bridge, outdir, msg, fmt=img_format)
                head,tail = os.path.split(results['filename'])
                results['filename']=tail
                # results['filename'] = os.path.relpath(results['filename'], yaml_outdir)
                camera2dict(msg, results, camera_dict_left,"left")
                stats['img_count'] += 1
                # stats['msg_count'] += 1

            if  (topic == CENTER_IMAGE_TOPIC):
                outdir = center_outdir
                if debug_print:
                    print("%s_camera %d" % (topic[1],timestamp))
                results = write_image(bridge, outdir, msg, fmt=img_format)
                head,tail = os.path.split(results['filename'])
                results['filename']=tail
                # results['filename'] = os.path.relpath(results['filename'], yaml_outdir)
                camera2dict(msg, results, camera_dict_center,"center")
                stats['img_count'] += 1
                # stats['msg_count'] += 1

            if  (topic == RIGHT_IMAGE_TOPIC):
                outdir = right_outdir
                if debug_print:
                    print("%s_camera %d" % (topic[1],timestamp))
                results = write_image(bridge, outdir, msg, fmt=img_format)
                head,tail = os.path.split(results['filename'])
                results['filename']=tail
                # results['filename'] = os.path.relpath(results['filename'], yaml_outdir)
                camera2dict(msg, results, camera_dict_right, "right")
                stats['img_count'] += 1
                # stats['msg_count'] += 1

            elif topic == STEERING_TOPIC:
                if debug_print:
                   print("steering %d %f" % (timestamp, msg.data))
                steering2dict(msg, steering_dict)
                stats['msg_count'] += 1

        # no need to cycle through readers in any order for dumping, rip through each on in sequence
        for reader in readers:
            for result in reader.read_messages():
                _process_msg(*result, stats=stats_acc)
                if ((stats_acc['img_count'] and stats_acc['img_count'] % 1000 == 0) or
                        (stats_acc['msg_count'] and stats_acc['msg_count'] % 10000 == 0)):
                    print("%d images, %d messages processed..." %
                          (stats_acc['img_count'], stats_acc['msg_count']))
                    sys.stdout.flush()

        print("Writing done. %d images, %d messages processed." %
              (stats_acc['img_count'], stats_acc['msg_count']))
        sys.stdout.flush()

        if include_images:
            camera_center_df = pd.DataFrame(data=camera_dict_center , columns=["timestamp", "center"])
            camera_left_df = pd.DataFrame(data=camera_dict_left , columns=["left"])
            camera_right_df = pd.DataFrame(data=camera_dict_right , columns=["right"])

        steering_df = pd.DataFrame(data=steering_dict, columns=steering_cols)
        data_df = pd.DataFrame(data=None , columns=[ "timestamp", "center", "left", "right", "angle"])
        data_df['timestamp'] = camera_center_df['timestamp']
        data_df['center'] = camera_center_df['center']
        data_df['left'] = camera_left_df['left']
        data_df['right'] = camera_right_df['right']
        data_df['angle'] = steering_df['angle']
        data_csv_path = os.path.join(yaml_outdir, 'data_lcr.csv')
        data_df.to_csv(data_csv_path, index=False)



if __name__ == '__main__':
    main()
