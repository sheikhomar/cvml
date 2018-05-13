from model_base import ModelBase

from keras import applications
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, MaxPooling2D
from keras.applications.imagenet_utils import preprocess_input


def preprocessor(x):
    return preprocess_input(x)


class ModelVGG16Take3(ModelBase):
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
        x = base_model.output
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.n_labels, activation='softmax')(x)

        self.model = Model(inputs=base_model.input, outputs=x)


if __name__ == '__main__':
    ModelVGG16Take3(
        learning_rate=None,
        optimizer=optimizers.SGD(momentum=0.9),
        batch_size=64,
        preprocessor=preprocessor,
        verbose=10
    ).train()
