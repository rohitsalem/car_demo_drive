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


def get_outdir(base_dir, name):
    outdir = os.path.join(base_dir, name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

def write_image(bridge, outdir, msg, fmt='png'):
    results = {}
    random_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in xrange(10)])
    # image_filename = os.path.join(outdir, str(msg.header.stamp.to_nsec()) + '.' + fmt)
    image_filename = os.path.join(outdir, str(random_name) + '.' + fmt)

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


def camera2dict(msg, write_results, camera_dict):
    camera_dict["timestamp"].append(msg.header.stamp.to_nsec())
    camera_dict["width"].append(write_results['width'] if 'width' in write_results else msg.width)
    camera_dict['height'].append(write_results['height'] if 'height' in write_results else msg.height)
    camera_dict["frame_id"].append(msg.header.frame_id)
    camera_dict["filename"].append(write_results['filename'])


def steering2dict(msg, steering_dict):
    steering_dict["timestamp"].append(msg.header.stamp.to_nsec())
    steering_dict["angle"].append(msg.data)
   

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

    STEERING_TOPIC="/filtered/steering_angle"
    CAMERA_TOPICS="/filtered/center/image_raw"

    filter_topics = [STEERING_TOPIC , CAMERA_TOPICS]
    

    bagsets = find_bagsets(data_dir, filter_topics=filter_topics)
    for bs in bagsets:
        print("Processing set %s" % bs.name)
        sys.stdout.flush()

        dataset_outdir = os.path.join(data_dir, "%s" % bs.name)
        center_outdir = get_outdir(data_dir, "center")
        yaml_outdir = get_outdir(data_dir, "yaml_files")

        camera_cols = ["timestamp", "width", "height", "frame_id", "filename"]
        camera_dict = defaultdict(list)

        steering_cols = ["timestamp", "angle"]
        steering_dict = defaultdict(list)

        bs.write_infos(yaml_outdir)
        readers = bs.get_readers()
        stats_acc = defaultdict(int)

        def _process_msg(topic, msg, stats):
            timestamp = msg.header.stamp.to_nsec()
            if topic in CAMERA_TOPICS:
                outdir = center_outdir
                if debug_print:
                    print("%s_camera %d" % (topic[1],timestamp))

                results = write_image(bridge, outdir, msg, fmt=img_format)
                results['filename'] = os.path.relpath(results['filename'], yaml_outdir)
                camera2dict(msg, results, camera_dict)
                stats['img_count'] += 1
                stats['msg_count'] += 1

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
            camera_csv_path = os.path.join(yaml_outdir, 'camera.csv')
            camera_df = pd.DataFrame(data=camera_dict, columns=camera_cols)
            camera_df.to_csv(camera_csv_path, index=False)
            # camera_steering_csv_path = os.path.join(yaml_outdir, 'camera_steering.csv')
            # camera_steering_df = pd.DataFrame(data = camera_dict,steering_dict, columns = camera_cols,steering_cols)
            # camera_steering_df.to_csv(camera_steering_csv_path, index=False)


        steering_csv_path = os.path.join(yaml_outdir, 'steering.csv')
        steering_df = pd.DataFrame(data=steering_dict, columns=steering_cols)
        steering_df.to_csv(steering_csv_path, index=False)

        
        # if include_images and gen_interpolated:
        #     # A little pandas magic to interpolate steering/gps samples to camera frames
        #     camera_df['timestamp'] = pd.to_datetime(camera_df['timestamp'])
        #     camera_df.set_index(['timestamp'], inplace=True)
        #     camera_df.index.rename('index', inplace=True)
        #     steering_df['timestamp'] = pd.to_datetime(steering_df['timestamp'])
        #     steering_df.set_index(['timestamp'], inplace=True)
        #     steering_df.index.rename('index', inplace=True)
        #     # gps_df['timestamp'] = pd.to_datetime(gps_df['timestamp'])
        #     # gps_df.set_index(['timestamp'], inplace=True)
        #     # gps_df.index.rename('index', inplace=True)

        #     merged = functools.reduce(lambda left, right: pd.merge(left, right, how='outer', left_index=True, right_index=True), [camera_df, steering_df])
        #     merged.interpolate(method='time', inplace=True)

        #     filtered_cols = ['timestamp', 'width', 'height', 'frame_id', 'filename']
                             
        #     filtered = merged.loc[camera_df.index]  # back to only camera rows
        #     filtered.fillna(0.0, inplace=True)
        #     filtered['timestamp'] = filtered.index.astype('int')  # add back original timestamp integer col
        #     filtered['width'] = filtered['width'].astype('int')  # cast back to int
        #     filtered['height'] = filtered['height'].astype('int')  # cast back to int
        #     filtered = filtered[filtered_cols]  # filter and reorder columns for final output

        #     interpolated_csv_path = os.path.join(yaml_outdir, 'final_interpolated.csv')
        #     filtered.to_csv(interpolated_csv_path, header=True)

        #     # process the images and generate the independent csv files for cameras
        #     image_csv = pd.read_csv(interpolated_csv_path, sep=',', header=0)

        #     center_csv = image_csv[image_csv['frame_id'] == 'camera_link']
        #     c_outdir = os.path.join(yaml_outdir, 'center_camera_image.csv')
        #     center_csv.to_csv(c_outdir, index=False, sep=',')

if __name__ == '__main__':
    main()
