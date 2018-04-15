from model_base import ModelBase

from keras import applications
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense


class ModelVGG16(ModelBase):
    def __init__(self, *args, **kwargs):
        ModelBase.__init__(self, *args, **kwargs)
        self.imagenet_weights_url = \
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.1' \
            '/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

    def _create(self):
        base_model = applications.VGG16(
            weights=None,
            include_top=False,
            input_shape=(self.img_width, self.img_height, self.img_channels)
        )
        self.model = Sequential(base_model.layers)
        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(self.n_labels, activation='softmax'))
