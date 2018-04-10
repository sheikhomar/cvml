import os
import re
import sys
import urllib.request
import h5py

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras import optimizers


class ModelBase:
    def __init__(self, model_name=None, data_augmentation_factor=5, batch_size=16, verbose=0):
        if model_name is None:
            script_name, script_ext = os.path.splitext(sys.argv[0])
            self.model_name = script_name
        else:
            self.model_name = model_name
        self.data_augmentation_factor = data_augmentation_factor
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
        self.epochs = 10

    def find_best_model(self, models_dir='./models/'):
        list_of_files = sorted(os.listdir(models_dir))
        best_model = None
        best_acc = 0
        best_loss = float('inf')
        for f in list_of_files:
            if f.startswith(self.model_name):
                values = re.findall('val[^\d]*(\d+\.\d*)', f)
                acc = float(values[0])
                loss = float(values[1])
                if acc > best_acc and loss < best_loss:
                    best_acc = acc
                    best_loss = loss
                    best_model = os.path.join(models_dir, f)
        return best_model

    def find_prev_best_model(self):
        saved_model_path = self.find_best_model()
        if saved_model_path is not None:
            return load_model(saved_model_path)
        return None

    def train_model(self, model, learning_rate=0.001):
        model.compile(
            optimizers.Adam(lr=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Print out final model
        print(model.summary())

        # Define model checkpoint
        checkpoint = ModelCheckpoint(
            'models/%s-epoch{epoch:02d}-valacc{val_acc:.2f}-valloss{val_loss:.2f}-acc{acc:.2f}-loss{loss:.2f}.hdf5' % self.model_name,
            monitor='val_acc',
            save_best_only=False,
            save_weights_only=True,
            mode='auto',
            period=1,
            verbose=self.verbose
        )

        # Define early stopping
        early = EarlyStopping(
            monitor='val_acc',
            min_delta=0,
            patience=10,
            mode='auto',
            verbose=self.verbose
        )

        # Data generators for the model
        train_batches = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=True,
            fill_mode="nearest",
            zoom_range=0.3,
            width_shift_range=0.3,
            height_shift_range=0.3,
            rotation_range=30
        ).flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode="categorical"
        )

        validation_batches = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=True,
            fill_mode="nearest",
            zoom_range=0.3,
            width_shift_range=0.3,
            height_shift_range=0.3,
            rotation_range=30
        ).flow_from_directory(
            self.validation_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode="categorical"
        )

        print('Training model...')

        model.fit_generator(
            train_batches,
            steps_per_epoch=int(self.n_train_samples / self.batch_size) * self.data_augmentation_factor,
            validation_data=validation_batches,
            validation_steps=int(self.n_validation_samples / self.batch_size) * self.data_augmentation_factor,
            epochs=self.epochs,
            callbacks=[checkpoint, early],
            verbose=self.verbose,
        )

    def load_pretrained_weights(self, model, model_weights_url):
        model_weights_path = 'models/{}'.format(os.path.basename(model_weights_url))
        if os.path.isfile(model_weights_path):
            print('Model file already downloaded')
        else:
            # Download pre-trained weights
            print('Downloading {}...'.format(model_weights_path))
            urllib.request.urlretrieve(model_weights_url, model_weights_path)

        print('Loading weights...')
        # Load weights from the downloaded file
        with h5py.File(model_weights_path) as model_weights_file:
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
                model.layers[i].set_weights(transferred_weights)
        print('Done loading weights')
