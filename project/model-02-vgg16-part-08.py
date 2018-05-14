from keras.optimizers import SGD

from model_base import ModelBase

from keras import applications
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense


class ModelVGG16(ModelBase):
    def __init__(self, *args, **kwargs):
        ModelBase.__init__(self, *args, **kwargs)

    def _create(self):
        base_model = applications.VGG16(
            weights=None,
            include_top=False,
            input_shape=(self.img_width, self.img_height, self.img_channels)
        )
        self.model = Sequential(base_model.layers)
        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dense(self.n_labels, activation='softmax'))


ModelVGG16(
  learning_rate=None,
  optimizer=SGD(momentum=0.9),
  n_freeze_layers=0,
  batch_size=64
).train()
