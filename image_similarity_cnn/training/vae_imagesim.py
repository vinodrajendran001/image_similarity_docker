from keras.layers import Input, Dense, Lambda, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras import objectives
import numpy as np

import matplotlib
import os
from numpy import *
import cv2
import h5py
import matplotlib.pyplot as plt
# SKLEARN
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
data_augmentation = True

#constant definition
batch_size = 2
original_dim = 2352
latent_dim = 10
intermediate_dim = 256
nb_epoch = 500
epsilon_std = 1.0

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              std=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

def deep(x_train, x_test):

    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    vae = Model(x, x_decoded_mean)
    vae.compile(optimizer='rmsprop', loss=vae_loss)
    checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)

    vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
        callbacks=[TensorBoard(log_dir='/tmp/vae'),
                                             checkpointer,
                                             EarlyStopping(monitor='val_loss', patience=10, verbose=0)
                                             ])

    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)

    return encoder

def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

def load_data(file):
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

    data, Label = shuffle(immatrix, label, random_state=2)
    train_data = [data, Label]

    return train_data


if __name__ == "__main__":

    listing = os.listdir(path1)
    num_samples = size(listing)
    print (num_samples)

    for file in listing:
        data = load_data(file)

    (X, Y) = (data[0], data[1])

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.335, random_state=4)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    model_act = deep(x_train, x_test)

    model_act.save('vaemodel.h5')

    x_test_encoded = model_act.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    plt.colorbar()
    plt.show()

