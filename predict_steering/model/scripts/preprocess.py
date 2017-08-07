
import cv2
import os
import numpy as np
import matplotlib.image as mpimg

input_shape = (64,64, 3)
img_height, img_width, channels = input_shape

def rgb2yuv(image):
    # Convert image from RGB to YUV as done by NVIDIA
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def load_image(data_dir, image_file):
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))

def resize(image):
    return cv2.resize(image, (img_width, img_height), cv2.INTER_CUBIC)

def crop(image):
    return image[60:-25, :, :]

def preprocess(image):
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image

# def choose_image(data_dir, center, left, right, steering_angle):
#     choice = np.random.choice(3)
#     if choice == 0:
#         return load_image(data_dir, left), steering_angle + 0.2
#     elif choice == 1:
#         return load_image(data_dir, right), steering_angle - 0.2
#     return load_image(data_dir, center), steering_angle

def random_flip(image, steering_angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def augument(data_dir, center, steering_angle, range_x=100, range_y=10):
    image, steering_angle = load_image(data_dir, center), steering_angle
    image, steering_angle = random_flip(image, steering_angle)
    image = random_brightness(image)
    return image, steering_angle

def data_iterator(data_dir, image_path, steering_angles, batch_size, is_training):
    images = np.empty([batch_size, img_height, img_width, channels])
    steerings = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_path.shape[0]):
            center= image_path[index]
            steering_angle = steering_angles[index]
            # Random argumentation with 50% probability
            if is_training and np.random.rand() < 0.5:
                image, steering_angle = augument(data_dir, center, steering_angle)
            else:
                image = load_image(data_dir, center)
            images[i] = preprocess(image)
            steerings[i] = steering_angle
            i += 1
            if i == batch_size:
                yield images, steerings
                break
