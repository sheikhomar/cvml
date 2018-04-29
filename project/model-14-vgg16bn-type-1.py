from model_base import ModelBase

import numpy as np

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Dense, Lambda, Conv2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D

class ModelVGG16BN(ModelBase):
    def __init__(self, *args, **kwargs):
        ModelBase.__init__(self, *args, **kwargs)

    def _add_convolution_block(self, filters, layers):
        self.block_no += 1

        for layer in range(1, layers + 1):
            conv_name = 'block{}_conv{}'.format(self.block_no, layer)
            bn_name = 'block{}_bn{}'.format(self.block_no, layer)
            act_name = 'block{}_activation{}'.format(self.block_no, layer)
            if self.block_no == 1 and layer == 1:
                input_shape = (self.img_width, self.img_height, self.img_channels)
                self.model.add(Conv2D(filters, (3, 3), padding='same', name=conv_name, input_shape=input_shape))
            else:
                self.model.add(Conv2D(filters, (3, 3), padding='same', name=conv_name))
            self.model.add(BatchNormalization(name=bn_name))
            self.model.add(Activation('relu', name=act_name))

        pool_name = 'block{}_pool'.format(self.block_no)
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2), name=pool_name))

    def _add_fully_connected_block(self, name, dropout=0.5):
        self.model.add(Dense(4096, name=name))
        self.model.add(BatchNormalization(name=name + '_bn'))
        self.model.add(Activation('relu', name=name + '_activation'))
        self.model.add(Dropout(dropout, name=name + '_dropout'))

    def _create(self):
        self.model = Sequential()
        self.block_no = 0

        self._add_convolution_block(filters=64,  layers=2)
        self._add_convolution_block(filters=128, layers=2)
        self._add_convolution_block(filters=256, layers=3)

        self.model.add(Flatten())

        self._add_fully_connected_block(name='fc1', dropout=0.5)
        self._add_fully_connected_block(name='fc2', dropout=0.2)

        self.model.add(Dense(self.n_labels, activation='softmax', name='output'))


ModelVGG16BN(
    learning_rate=0.0001,
    batch_size=128
).train()
