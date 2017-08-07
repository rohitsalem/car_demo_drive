#!/usr/bin/python
## Author: sumanth
## Date: Feb, 05,2017
# test file to show the processed images
import matplotlib.pyplot as plt
import numpy as np
import processData
import pandas as pd

id =17
data = pd.read_csv(processData.dataPath)
img = plt.imread(processData.imPath+data.iloc[id]['center'].strip())
angle = data.iloc[id]['steering']
print('angle')

#indivudaul images
plt.imshow(plt.imread(processData.imPath+data.iloc[id]['center'].strip()))
plt.title("center")
plt.savefig('center.png')
plt.show(block=True)

# plt.imshow(plt.imread(processData.imPath+data.iloc[id]['left'].strip()))
# plt.title("left")
# plt.savefig('left.png')
# plt.show(block=True)

# plt.imshow(plt.imread(processData.imPath+data.iloc[id]['right'].strip()))
# plt.title("right")
# plt.savefig('right.png')
# plt.show(block=True)

# shear image
image, steering_angle = processData.randomShear(img, angle)
plt.imshow(image)
plt.title("shear")
plt.savefig('shear.png')
plt.show(block=True)

# crop image
imagecr = processData.crop(img, 0.3, 0.27)
plt.imshow(image)
plt.title("crop")
plt.savefig('crop.png')
plt.show(block=True)

# flip the image
image, steering_angle = processData.randomFlip(img, angle)
plt.imshow(image)
plt.title("flip")
plt.savefig('flip.png')
plt.show(block=True)

# random gamma
# image = processData.randomGamma(img)
# plt.imshow(image)
# plt.title("gamma")
# plt.savefig('gamma.png')
# plt.show(block=True)

# resize
image = processData.resize(imagecr, (64,64))
plt.imshow(image)
plt.title("resize")
plt.savefig('resize.png')
plt.show(block=True)

# all techniques to asingle image
image, steering_angle = processData.randomShear(img, angle)
plt.imshow(image)
plt.title("shear1")
plt.savefig('shear1.png')
plt.show(block=True)

# crop image
imagecrr = processData.crop(image, 0.3, 0.27)
plt.imshow(image)
plt.title("crop2")
plt.savefig('crop2.png')
plt.show(block=True)

# flip the image
image, steering_angle = processData.randomFlip(image, angle)
plt.imshow(image)
plt.title("fli3")
plt.savefig('flip3.png')
plt.show(block=True)

# random gamma
# image = processData.randomGamma(image)
# plt.imshow(image)
# plt.title("gamma4")
# plt.savefig('gamma4.png')
# plt.show(block=True)

# resize
image = processData.resize(imagecrr, (64,64))
plt.imshow(image)
plt.title("resize5")
plt.savefig('resize5.png')
plt.show(block=True)
