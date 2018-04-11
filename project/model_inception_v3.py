from model_base import ModelBase

from keras import applications
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Dense, GlobalAveragePooling2D


class ModelInceptionV3(ModelBase):
    def __init__(self):
        ModelBase.__init__(self)

    def get_model(self, n_frozen_layers=0):
        base_model = applications.InceptionV3(
            weights=None,
            include_top=False,
            input_shape=(self.img_width, self.img_height, self.img_channels)
        )

        if n_frozen_layers > 1:
            for layer in base_model.layers[1:n_frozen_layers]:
                layer.trainable = False

        output_layer = base_model.output
        output_layer = GlobalAveragePooling2D(name='avg_pool')(output_layer)
        output_layer = Dropout(0.5)(output_layer)
        output_layer = Dense(self.n_labels, activation='softmax', name='predictions')(output_layer)

        new_model = Model(inputs=base_model.input, outputs=output_layer)

        self.load_saved_weights(new_model)

        return new_model

    def run(self, learning_rate, n_frozen_layers, batch_size=4, verbose=1):
        self.batch_size = batch_size
        self.verbose = verbose
        m = self.get_model(n_frozen_layers=n_frozen_layers)
        self.train_model(m, learning_rate=learning_rate)
