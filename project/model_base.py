import os
import re
import sys
import urllib.request
import h5py

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.models import Sequential
from keras import optimizers

import numpy as np


class ModelBase:
    def __init__(self,
                 model_name=None,
                 batch_size=16,
                 verbose=0,
                 n_freeze_layers=0,
                 learning_rate=0.00001,
                 epochs=400,
                 preprocessor=None,
                 optimizer=None
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
        self.epochs = epochs
        self.n_freeze_layers = n_freeze_layers
        self.learning_rate = learning_rate
        self.imagenet_weights_url = None
        self.imagenet_use_id = False
        self.preprocessor = preprocessor

        self.optimizer = optimizer
        if self.optimizer is not None and self.learning_rate is not None:
            raise Exception('Optimizer and learning cannot be set at the same time.')
        if self.learning_rate is None:
            self.learning_rate = 1e-3
        if self.optimizer is None:
            self.optimizer = optimizers.Adam(lr=self.learning_rate)

    def load_model(self, model_weights=None):
        print('Creating model...')
        self._create()

        print('Loading weights from {}...'.format(model_weights))
        self.model.load_weights(model_weights)

        print('Compiling using optimizer: %s...' % self.optimizer)
        self.model.compile(
            self.optimizer,
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

        print('Compiling using optimizer: %s...' % self.optimizer)
        self.model.compile(
            self.optimizer,
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

    def predict_validation(self, model_weights=None):
        img_paths = self._get_validation_image_paths()
        return self._predict(model_weights, img_paths)

    def predict_test(self, model_weights=None):
        img_paths = self._get_test_image_paths()
        return self._predict(model_weights, img_paths)

    def predict_instance_validation(self, model_weights=None):
        img_paths = self._get_validation_image_paths()
        return self._predict_instance(model_weights, img_paths)

    def predict_instance_test(self, model_weights=None):
        img_paths = self._get_test_image_paths()
        return self._predict_instance(model_weights, img_paths)

    @staticmethod
    def write_predictions(predictions, file_name='predictions.csv'):
        with open(file_name, 'w') as file:
            file.write('ID,Label')
            for index, value in enumerate(predictions):
                file.write('\n{0},{1}'.format(index + 1, value))

    @staticmethod
    def show_progress_bar(iteration, total, bar_length=50):
        percent = int(round((iteration / total) * 100))
        nb_bar_fill = int(round((bar_length * percent) / 100))
        bar_fill = '#' * nb_bar_fill
        bar_empty = ' ' * (bar_length - nb_bar_fill)
        sys.stdout.write("\r  [{0}] {1}%".format(str(bar_fill + bar_empty), percent))
        sys.stdout.flush()

    def _predict(self, model_weights, img_paths):
        if model_weights is None:
            model_weights = self._find_saved_weights()
        if model_weights is None:
            raise Exception('Please provide a path to the model weights to use!')

        self.load_model(model_weights)

        label_map = self._get_label_map()
        img_count = len(img_paths)
        y_predictions = np.zeros(img_count, dtype=np.int8)

        for i, (img_num, img_path) in enumerate(img_paths):
            ModelBase.show_progress_bar(i, img_count)
            img_data = self._load_image(img_path)
            if isinstance(self.model, Sequential):
                pred_index = self.model.predict_classes(img_data)[0]
            else:
                predictions = self.model.predict(img_data)
                pred_index = np.argmax(predictions, axis=1)[0]
            pred_label = label_map[pred_index]
            y_predictions[img_num-1] = pred_label

        return y_predictions

    def _predict_instance(self, model_weights, img_paths):
        if model_weights is None:
            model_weights = self._find_saved_weights()
        if model_weights is None:
            raise Exception('Please provide a path to the model weights to use!')

        print('Instance-based predictions...')
        self.load_model(model_weights)

        # Sort image paths in-place
        img_paths.sort(key=lambda tup: tup[0])

        # Group images so they come in pairs
        instance_pairs = list(zip(*[iter(img_paths)] * 2))

        label_map = self._get_label_map()
        img_count = len(img_paths)
        y_predictions = np.zeros(img_count, dtype=np.int8)

        for i, ((img1_num, img1_path), (img2_num, img2_path)) in enumerate(instance_pairs):
            ModelBase.show_progress_bar(i, img_count)

            img1_data = self._load_image(img1_path)
            img2_data = self._load_image(img2_path)
            img1_pred = self.model.predict(img1_data)
            img2_pred = self.model.predict(img2_data)

            img1_pred_index = np.argmax(img1_pred, axis=1)[0]
            img2_pred_index = np.argmax(img2_pred, axis=1)[0]

            if img1_pred_index != img2_pred_index:
                img1_highest_score = np.max(img1_pred, axis=1)[0]
                img2_highest_score = np.max(img2_pred, axis=1)[0]

                # if class labels for different views differ,
                # we assign to the instance the class label
                # with the highest confidence score.
                if img1_highest_score > img2_highest_score:
                    img2_pred_index = img1_pred_index
                else:
                    img1_pred_index = img2_pred_index

            img1_pred_label = label_map[img1_pred_index]
            img2_pred_label = label_map[img2_pred_index]

            y_predictions[img1_num - 1] = img1_pred_label
            y_predictions[img2_num - 1] = img2_pred_label

        print('\n ... predictions done')
        return y_predictions

    def _get_label_map(self):
        # We need the ImageDataGenerator used to train the model
        # because it contains a mapping between classes and indices
        train_gen = self._get_train_generator()

        # Reverse keys and values so values becomes keys
        label_map = {v: int(k) for k, v in train_gen.class_indices.items()}

        return label_map

    def _load_image(self, image_path):
        img = image.load_img(image_path, target_size=(self.img_width, self.img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        if self.preprocessor is None:
            return x
        return self.preprocessor(x)

    def _get_test_image_paths(self):
        final_list = []
        for img_name in os.listdir(self.test_data_dir):
            img_number = int(re.findall(r'\d+', img_name)[0])
            img_path = os.path.join(self.test_data_dir, img_name)
            final_list.append((img_number, img_path))
        return final_list

    def _get_validation_image_paths(self):
        final_list = []
        for sub_dir in os.listdir(self.validation_data_dir):
            sub_dir_path = os.path.join(self.validation_data_dir, sub_dir)
            for img_name in os.listdir(sub_dir_path):
                img_number = int(re.findall(r'\d+', img_name)[0])
                img_path = os.path.join(sub_dir_path, img_name)
                final_list.append((img_number, img_path))
        return final_list

    def _get_validation_generator(self):
        return ImageDataGenerator(
            preprocessing_function=self.preprocessor
        ).flow_from_directory(
            self.validation_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode="categorical"
        )

    def _get_train_generator(self):
        return ImageDataGenerator(
            preprocessing_function=self.preprocessor
        ).flow_from_directory(
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
            'saved_weights/%s-epoch{epoch:02d}-acc{acc:.4f}-loss{loss:.4f}'
            '-valacc{val_acc:.4f}-valloss{val_loss:.4f}.hdf5' % self.model_name,
            monitor='val_acc',
            save_best_only=True,
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

        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-5
        )

        # Log epoch history
        logger = CSVLogger(
            filename='logs/%s.csv' % self.model_name,
            append=True
        )

        return [checkpoint, early_stop, reduce_lr, logger]

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

        layer_indices = {l.name: i for (i, l) in enumerate(self.model.layers)}

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
                if self.imagenet_use_id:
                    layer_index = i
                else:
                    layer_key = layer_name.decode('UTF-8')
                    if layer_key not in layer_indices:
                        continue
                    layer_index = layer_indices[layer_key]
                self.model.layers[layer_index].set_weights(transferred_weights)
        print('Done loading weights')

    def _preprocess_input(self, x):
        if self.preprocessor is not None:
            self.preprocessor(x)
