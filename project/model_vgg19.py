from model_base import ModelBase

from keras import applications
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, Conv2D


class ModelVGG19(ModelBase):
    def __init__(self):
        ModelBase.__init__(self)

    def get_model(self, n_frozen_layers=0):
        new_model = self.find_prev_best_model()
        if new_model is None:
            base_model = applications.VGG19(
                weights=None,
                include_top=False,
                input_shape=(self.img_width, self.img_height, self.img_channels)
            )
            url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
            self.load_pretrained_weights(base_model, url)

            if n_frozen_layers > 1:
                for layer in base_model.layers[1:n_frozen_layers]:
                    layer.trainable = False

            new_model = Sequential(base_model.layers)
            new_model.add(Flatten())
            new_model.add(Dense(1024, activation='relu'))
            new_model.add(Dropout(0.5))
            new_model.add(Dense(1024, activation='relu'))
            new_model.add(Dense(self.n_labels, activation='softmax'))

        return new_model

    def run(self, learning_rate, n_frozen_layers, batch_size=16):
        self.batch_size = batch_size
        m = self.get_model(n_frozen_layers=n_frozen_layers)
        self.train_model(m, learning_rate=learning_rate)
