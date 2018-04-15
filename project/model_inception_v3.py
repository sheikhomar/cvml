from model_base import ModelBase

from keras import applications
from keras.models import Model
from keras.layers import Dropout, Dense, GlobalAveragePooling2D


class ModelInceptionV3(ModelBase):
    def __init__(self, *args, **kwargs):
        ModelBase.__init__(self, *args, **kwargs)
        self.imagenet_use_id = True
        self.imagenet_weights_url = \
            'https://github.com/fchollet/deep-learning-models/releases/download' \
            '/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

    def _create(self):
        base_model = applications.InceptionV3(
            weights=None,
            include_top=False,
            input_shape=(self.img_width, self.img_height, self.img_channels)
        )
        output_layer = base_model.output
        output_layer = GlobalAveragePooling2D(name='avg_pool')(output_layer)
        output_layer = Dropout(0.5)(output_layer)
        output_layer = Dense(self.n_labels, activation='softmax', name='predictions')(output_layer)
        self.model = Model(inputs=base_model.input, outputs=output_layer)
