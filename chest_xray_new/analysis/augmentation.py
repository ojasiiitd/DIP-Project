# import os
# import cv2
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense , Conv2D , MaxPooling2D , AveragePooling2D , Flatten , Dropout , BatchNormalization
# from keras.preprocessing.image import ImageDataGenerator , load_img , img_to_array
# from keras import backend

# if __name__ == '__main__':
#     image = cv2.imread("train/PNEUMONIA/person1405_bacteria_3573.jpeg" )

#     x_train = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

#     datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=30,
#     zoom_range=0.2,
#     shear_range=0.2)

#     datagen.fit(x_train)
#     it = datagen.flow(x_train)

#     # fig, rows = plt.subplots(nrows=1, ncols=4, figsize=(18,18))
#     for i in range(4):
#         plt.imshow(it.next()[0].astype('int'))
#         plt.show()
#     # plt.show()

import cv2
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load the image
img = cv2.imread("../train/PNEUMONIA/person1405_bacteria_3573.jpeg" )
# img = load_img('train/PNEUMONIA/person1405_bacteria_3573.jpeg')
# convert to numpy array
data = img.copy()
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # generate batch of images
    batch = it.next()
    # convert to unsigned integers for viewing
    image = batch[0].astype('uint8')
    # plot raw pixel data
    pyplot.imshow(image)
# show the figure
pyplot.show()