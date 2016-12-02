#import
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

import matplotlib
import os
from numpy import *
import cv2
import h5py
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.callbacks import TensorBoard

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

dir_name = os.path.join(APP_ROOT, 'input/')

resize = os.path.join(APP_ROOT, 'input_resize/')

#Give the path of dataset
#path1 = '/home/vinod/PycharmProjects/image_upload/input/'  #path of folder of images
path1 = dir_name
#Give the path to store rescaled images
#path2 = '/home/vinod/PycharmProjects/image_upload/input_resized/'  #path of folder to save images
path2 = resize
#set true or false for data augmentation
data_augmentation = True

def deep(x_train, x_test):

    input_img = Input(shape=(28, 28, 3))
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)

    # at this point the representation is (8, 4, 4) i.e. 128-dimensional

    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(16, 3, 3, activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)

    if not data_augmentation:
        print('Not using data augmentation.')

        autoencoder.fit(x_train, x_train,
                        nb_epoch=500,
                        batch_size=10,
                        shuffle=True,
                        validation_data=(x_test, x_test),
                        callbacks=[TensorBoard(log_dir='/tmp/cae'),
                                   checkpointer,
                                   EarlyStopping(monitor='val_loss', patience=3, verbose=0)
                                   ])
    else:
        print('Using real-time data augmentation.')

        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(x_train)

        # fit the model on the batches generated by datagen.flow()
        autoencoder.fit_generator(datagen.flow(x_train, x_train, batch_size=10),
                                  samples_per_epoch=x_train.shape[0],
                                  nb_epoch=500,
                                  validation_data=(x_test, x_test),
                                  callbacks=[TensorBoard(log_dir='/tmp/cae_dataaug'),
                                             checkpointer,
                                             EarlyStopping(monitor='val_loss', patience=3, verbose=0)
                                             ])

    encoder = Model(input_img, encoded)

    return encoder

def load(file):
    # input image dimensions
    img_rows, img_cols = 28, 28

    # number of channels
    img_channels = 3

    image = cv2.imread(path1+file)
    img = cv2.resize(image, (img_rows, img_cols))
    cv2.imwrite(path2+file, img)

    imlist = os.listdir(path2)

    im1 = array(cv2.imread(path2+imlist[0])) # open one image to get size

    # format the data set suitable for cnn input
    immatrix = array([array(cv2.imread(path2+im2)) for im2 in imlist])

    # print (immatrix.shape)

    # 2 classes (not required just for ref)
    label=np.ones((num_samples,),dtype = int)
    label[0:94]=0
    label[94:]=1

    data, Label = shuffle(immatrix,label, random_state=2)
    train_data = [data, Label]

    return train_data


if __name__ == "__main__":

    listing = os.listdir(path1)
    num_samples = size(listing)
    print (num_samples)

    for file in listing:
        data = load(file)

    (X, Y) = (data[0], data[1])

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    model_act = deep(x_train, x_test)

    model_act.save('model_caedataaug.h5')

