import os
import re
import sys
import urllib.request
import h5py

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers


class ModelBase:
    def __init__(self,
                 model_name=None,
                 batch_size=16,
                 verbose=0,
                 n_freeze_layers=0,
                 learning_rate=0.00001
                 ):
        if model_name is None:
            script_name, script_ext = os.path.splitext(sys.argv[0])
            self.model_name = script_name
        else:
            self.model_name = model_name
        self.batch_size = batch_size
        self.verbose = verbose

        self.train_data_dir = "Train/TrainImages"
        self.validation_data_dir = "Validation/ValidationImages"
        self.test_data_dir = "Test/TestImages"
        self.img_width = 256
        self.img_height = 256
        self.img_channels = 3
        self.n_train_samples = 5830
        self.n_validation_samples = 2298
        self.n_test_samples = 3460
        self.n_labels = 29
        self.epochs = 100
        self.n_freeze_layers = n_freeze_layers
        self.learning_rate = learning_rate
        self.imagenet_weights_url = None

    def load_model(self, model_weights=None):
        print('Creating model...')
        self._create()
        print('Loading weights from {}...'.format(model_weights))
        self.model.load_weights(model_weights)
        print('Compiling...')
        self.model.compile(
            optimizers.Adam(lr=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return self.model

    def train(self):
        print('Creating model...')
        self._create()

        self._freeze_top_layers()

        print('Loading weights...')
        self._load_pretrained_weights()

        print('Compiling...')
        self.model.compile(
            optimizers.Adam(lr=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print(self.model.summary())

        # Data generators for the model
        train_gen = self._get_train_generator()
        validation_gen = self._get_validation_generator()

        print('Training model...')
        self.model.fit_generator(
            train_gen,
            steps_per_epoch=int(self.n_train_samples / self.batch_size),
            validation_data=validation_gen,
            validation_steps=int(self.n_validation_samples / self.batch_size),
            epochs=self.epochs,
            callbacks=self._get_callbacks(),
            verbose=self.verbose
        )

    def _get_validation_generator(self):
        return ImageDataGenerator().flow_from_directory(
            self.validation_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode="categorical"
        )

    def _get_train_generator(self):
        return ImageDataGenerator().flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode="categorical"
        )

    def _load_pretrained_weights(self):
        saved_weights_path = self._find_saved_weights()
        if saved_weights_path is not None:
            print('Loading saved weights from: {}'.format(saved_weights_path))
            self.model.load_weights(saved_weights_path)

        elif self.imagenet_weights_url is not None and len(self.imagenet_weights_url) > 0:
            print('Loading imagenet weights...')
            model_weights_path = 'saved_weights/{}'.format(os.path.basename(self.imagenet_weights_url))
            if os.path.isfile(model_weights_path):
                print('Model file already downloaded')
            else:
                # Download pre-trained weights
                print('Downloading {}...'.format(model_weights_path))
                urllib.request.urlretrieve(self.imagenet_weights_url, model_weights_path)
            self._load_weights_from_file(model_weights_path)
        else:
            print('No pre-trained weights loaded!')

    def _create(self):
        raise NotImplementedError('subclasses must override _create()')

    def _freeze_top_layers(self):
        if self.n_freeze_layers > 1:
            print("Freezing {} layers".format(self.n_freeze_layers))
            for layer in self.model.layers[:self.n_freeze_layers]:
                layer.trainable = False
            for layer in self.model.layers[self.n_freeze_layers:]:
                layer.trainable = True

    def _get_callbacks(self):
        # Define model checkpoint
        checkpoint = ModelCheckpoint(
            'saved_weights/%s-epoch{epoch:02d}-acc{acc:.2f}-loss{loss:.2f}'
            '-valacc{val_acc:.2f}-valloss{val_loss:.2f}.hdf5' % self.model_name,
            monitor='val_acc',
            save_best_only=False,
            save_weights_only=True,
            mode='auto',
            period=1,
            verbose=self.verbose
        )

        # Define early stopping
        early_stop = EarlyStopping(
            monitor='val_acc',
            min_delta=0,
            patience=10,
            mode='auto',
            verbose=self.verbose
        )

        return [checkpoint, early_stop]

    def _find_saved_weights(self, models_dir='./saved_weights/'):
        if not os.path.isdir(models_dir):
            return None

        list_of_files = sorted(os.listdir(models_dir))
        best_model = None
        best_acc = 0
        for f in list_of_files:
            if f.startswith(self.model_name):
                values = re.findall('val[^\d]*(\d+\.\d*)', f)
                acc = float(values[0])
                if acc > best_acc:
                    best_acc = acc
                    best_model = os.path.join(models_dir, f)
        return best_model

    def _load_weights_from_file(self, file_path):
        print('Loading weights from {}...'.format(file_path))
        # Load weights from the downloaded file
        with h5py.File(file_path) as model_weights_file:
            layer_names = model_weights_file.attrs['layer_names']
            for i, layer_name in enumerate(layer_names):
                level_0 = model_weights_file[layer_name]
                transferred_weights = []
                for k0 in level_0.keys():
                    level_1 = level_0[k0]
                    if hasattr(level_1, 'keys'):
                        for k1 in level_1.keys():
                            transferred_weights.append(level_1[k1][()])
                    else:
                        transferred_weights.append(level_0[k0][()])
                self.model.layers[i].set_weights(transferred_weights)
        print('Done loading weights')
