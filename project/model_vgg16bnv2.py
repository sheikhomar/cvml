from model_base import ModelBase

import numpy as np

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Dense, Lambda, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D


class ModelVGG16BNv2(ModelBase):
    def __init__(self, *args, **kwargs):
        ModelBase.__init__(self, *args, **kwargs)

    def _add_convolution_block(self, layers, filters):
        for i in range(layers):
            self.model.add(ZeroPadding2D((1, 1)))
            self.model.add(Conv2D(filters, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    def _add_fully_connected_block(self):
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

    def _create(self):
        self.model = Sequential()

        shape = (self.img_width, self.img_height, self.img_channels)
        self.model.add(Input(shape=shape))

        self._add_convolution_block(2, 64)
        self._add_convolution_block(2, 128)
        self._add_convolution_block(3, 256)
        self._add_convolution_block(3, 512)
        self._add_convolution_block(3, 512)

        self.model.add(Flatten())

        self._add_fully_connected_block()
        self._add_fully_connected_block()

        self.model.add(Dense(self.n_labels, activation='softmax'))
