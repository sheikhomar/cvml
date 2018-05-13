from model_base import ModelBase

from keras import applications
from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, MaxPooling2D


vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))


def vgg_preprocess(x):
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr


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
        preprocessor=vgg_preprocess,
        verbose=10
    ).train()
