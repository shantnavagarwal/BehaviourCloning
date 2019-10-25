import tensorflow
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
import matplotlib.pyplot as plt
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Cropping2D
from keras.layers.core import K
from datetime import datetime
from keras.models import load_model
from keras import optimizers

OrigXTrain = np.memmap('UXTrain1.npy', mode='r', shape=(48216, 160, 320, 3)) # Used memory map instead of generator
OrigYTrain = np.load('UYTrain1.npy')

imgshape = OrigXTrain[0].shape


def plotimg():
    plt.imshow(OrigXTrain[200])
    plt.show()


# Used to give name while saving model
def dateandtime():
    time = datetime.now()
    time = str(time)
    mins = time.split('.')
    mins[0] = mins[0].replace(' ', '_')
    mins[0] = mins[0].replace(':', '-')
    return mins[0]


def nvidiadeepnet():
    init = 'glorot_normal'
    activation = 'relu'
    model = Sequential()

    # pre-processing
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: K.tf.image.resize_images(x, (66, 200))))  # resize image
    model.add(Lambda(lambda x: x / 255.0 - 0.5))  # normalization

    # Convolutions
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', init=init))
    model.add(Activation(activation))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', init=init))
    model.add(Activation(activation))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', init=init))
    model.add(Activation(activation))

    model.add(Convolution2D(64, 3, 3, init=init))
    model.add(Activation(activation))

    model.add(Convolution2D(64, 3, 3, init=init))
    model.add(Activation(activation))

    # Fully Connected
    model.add(Flatten())
    model.add(Dense(100, init=init))
    model.add(Dropout(0.75))
    model.add(Dense(50, init=init))
    model.add(Dense(10, init=init))
    model.add(Dense(1, init=init))

    # model.summary

    return model


def lenet():
    model = Sequential()

    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: K.tf.image.rgb_to_grayscale(x)))
    model.add(Lambda(lambda x: K.tf.image.resize_images(x, (32, 32))))  # resize image
    model.add(Lambda(lambda x: x / 255.0 - 0.5))  # normalization

    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    return model


def trainnewmodel():
    nmodel = nvidiadeepnet()
    nmodel.compile(loss='mse', optimizer='adam')
    nmodel.fit(OrigXTrain, OrigYTrain, validation_split=0.2, shuffle=True, epochs=3, batch_size=512)
    savename = dateandtime()
    nmodel.save(savename + 'model.h5')
    nmodel.save_weights(savename + 'nmodelweight.h5')


def trainexismodel():
    filename = input()
    nmodel = load_model(filename)
    adam = optimizers.adam(lr=0.00001)
    nmodel.compile(loss='mse', optimizer=adam)
    nmodel.fit(OrigXTrain, OrigYTrain, validation_split=0.2, shuffle=True, epochs=20, batch_size=64)
    savename = dateandtime()
    nmodel.save(savename + 'model.h5')
    nmodel.save_weights(savename + 'nmodelweight.h5')


trainnewmodel()
# trainexismodel()
