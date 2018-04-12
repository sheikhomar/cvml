from model_base import ModelBase

from keras.models import Sequential
from keras import layers


class ModelHomebrew(ModelBase):
    def __init__(self, *args, **kwargs):
        ModelBase.__init__(self, *args, **kwargs)

    def _create(self):
        self.model = Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), input_shape=(self.img_width, self.img_height, self.img_channels)))
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(layers.Conv2D(32, (3, 3)))
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(layers.Conv2D(64, (3, 3)))
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
        self.model.add(layers.Dense(64))
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(self.n_labels, activation='softmax', name='predictions'))
