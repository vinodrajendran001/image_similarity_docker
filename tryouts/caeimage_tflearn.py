from __future__ import (
    division,
    print_function,
    absolute_import
)
from six.moves import range

import numpy as np
import tflearn
from skimage import io

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from numpy import *
import cv2

# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split


#####
# input image dimensions
img_rows, img_cols = 32, 32

# number of channels
img_channels = 3

path1 = '/home/vinod/PycharmProjects/image_upload/input/'  #path of folder of images
path2 = '/home/vinod/PycharmProjects/image_upload/input_resized/'  #path of folder to save images

listing = os.listdir(path1)
num_samples=size(listing)
print (num_samples)

for file in listing:
    image = cv2.imread(path1+file)
    img = cv2.resize(image, (img_rows, img_cols))
    cv2.imwrite(path2+file, img)

imlist = os.listdir(path2)

im1 = array(cv2.imread(path2+imlist[0])) # open one image to get size
m, n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# format the data set suitable for cnn input
immatrix = array([array(cv2.imread(path2+im2)) for im2 in imlist])

print (immatrix.shape)

# 2 classes
label=np.ones((num_samples,),dtype = int)
label[0:84]=0
label[84:]=1

data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]

print (train_data[0].shape)
print (train_data[1].shape)

(X, Y) = (train_data[0],train_data[1])
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=4)

print('X_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

nb_feature = 64

side = x_train.shape[1]
channel = x_train.shape[3]

def encoder(inputs):
    net = tflearn.conv_2d(inputs, 16, 3, strides=2)
    net = tflearn.batch_normalization(net)
    net = tflearn.elu(net)

    net = tflearn.conv_2d(net, 16, 3, strides=1)
    net = tflearn.batch_normalization(net)
    net = tflearn.elu(net)

    net = tflearn.conv_2d(net, 32, 3, strides=2)
    net = tflearn.batch_normalization(net)
    net = tflearn.elu(net)

    net = tflearn.conv_2d(net, 32, 3, strides=1)
    net = tflearn.batch_normalization(net)
    net = tflearn.elu(net)

    net = tflearn.flatten(net)
    net = tflearn.fully_connected(net, nb_feature)
    net = tflearn.batch_normalization(net)
    net = tflearn.sigmoid(net)

    return net

def decoder(inputs):
    net = tflearn.fully_connected(inputs, (side // 2**2)**2 * 32, name='DecFC1')
    net = tflearn.batch_normalization(net, name='DecBN1')
    net = tflearn.elu(net)

    net = tflearn.reshape(net, (-1, side // 2**2, side // 2**2, 32))
    net = tflearn.conv_2d(net, 32, 3, name='DecConv1')
    net = tflearn.batch_normalization(net, name='DecBN2')
    net = tflearn.elu(net)

    net = tflearn.conv_2d_transpose(net, 16, 3, [side // 2, side // 2],
                                        strides=2, padding='same', name='DecConvT1')
    net = tflearn.batch_normalization(net, name='DecBN3')
    net = tflearn.elu(net)

    net = tflearn.conv_2d(net, 16, 3, name='DecConv2')
    net = tflearn.batch_normalization(net, name='DecBN4')
    net = tflearn.elu(net)

    net = tflearn.conv_2d_transpose(net, channel, 3, [side, side],
                                        strides=2, padding='same', activation='sigmoid',
                                        name='DecConvT2')

    return net

net = tflearn.input_data(shape=[None, side, side, channel])
net = encoder(net)
net = decoder(net)

net = tflearn.regression(net, optimizer='adam', loss='mean_square', metric=None)

model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='tensorboard/',
                    checkpoint_path='ckpt/')
model.fit(X, X, n_epoch=10, validation_set=(x_test, x_test),
          run_id="auto_encoder", batch_size=128)

print("\nSaving results after being encoded and decoded:")
testX = tflearn.data_utils.shuffle(x_test)[0]
decoded = model.predict(testX)

img = np.ndarray(shape=(side*10, side*10, channel))
for i in range(50):
    row = i // 10 * 2
    col = i % 10
    img[side*row:side*(row+1), side*col:side*(col+1), :] = testX[i]
    img[side*(row+1):side*(row+2), side*col:side*(col+1), :] = decoded[i]
img *= 255
img = img.astype(np.uint8)

io.imsave('decode.png', img)