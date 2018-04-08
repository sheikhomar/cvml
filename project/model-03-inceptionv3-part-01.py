import sys
import os
import time
import urllib.request
import h5py

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Conv2D, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

script_name, script_ext = os.path.splitext(sys.argv[0])

MODEL_NAME = script_name

TRAIN_DATA_DIR = "Train/TrainImages"
VALIDATION_DATA_DIR = "Validation/ValidationImages"
TEST_DATA_DIR = "Test/TestImages"

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

N_TRAIN_SAMPLES = 5830
N_VALIDATION_SAMPLES = 2298
N_TEST_SAMPLES = 3460
N_LABELS = 29

# Model parameters
DATA_AUGMENTATION_FACTOR = 5
N_FROZEN_LAYERS = 311
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 100
VERBOSE = 0

model_weights_url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
model_weights_path = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

print('Running {}'.format(MODEL_NAME))

# Generate model
inception_model = applications.InceptionV3(
  weights=None, 
  include_top=False, 
  input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
)

if (os.path.isfile(model_weights_path)):
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
    transfered_weights = []
    for k0 in level_0.keys():
      level_1 = level_0[k0]
      for k1 in level_1.keys():
        transfered_weights.append(level_1[k1][()])
    inception_model.layers[i].set_weights(transfered_weights)
print('Done loading weights')

# Freeze some bottom layers
for layer in inception_model.layers[1:N_FROZEN_LAYERS]:
  layer.trainable = False

# Add custom layers
output_layer = inception_model.output
output_layer = GlobalAveragePooling2D(name='avg_pool')(output_layer)
output_layer = Dropout(0.5)(output_layer)
output_layer = Dense(N_LABELS, activation='softmax', name='predictions')(output_layer)

model = Model(inputs=inception_model.input, outputs=output_layer)

model.compile(
  optimizers.Adam(lr=LEARNING_RATE), 
  loss='categorical_crossentropy', 
  metrics=['accuracy']
)

# Print out final model
print(model.summary())

# Define model checkpoint
checkpoint = ModelCheckpoint(
  'models/%s-epoch{epoch:02d}-valacc{val_acc:.2f}-valloss{val_loss:.2f}.hdf5' % MODEL_NAME,
  monitor='val_acc',
  save_best_only=False, 
  save_weights_only=True, 
  mode='auto', 
  period=1,
  verbose=VERBOSE
)

# Define early stopping
early = EarlyStopping(
  monitor='val_acc', 
  min_delta=0, 
  patience=10,
  mode='auto', 
  verbose=VERBOSE
)

# Data generators for the model
train_batches = ImageDataGenerator(
  rescale = 1./255,
  horizontal_flip = True,
  fill_mode = "nearest",
  zoom_range = 0.3,
  width_shift_range = 0.3,
  height_shift_range = 0.3,
  rotation_range = 30
).flow_from_directory(
  TRAIN_DATA_DIR,
  target_size = (IMG_HEIGHT, IMG_WIDTH),
  batch_size = BATCH_SIZE,
  class_mode = "categorical"
)

validation_batches = ImageDataGenerator(
  rescale = 1./255,
  horizontal_flip = True,
  fill_mode = "nearest",
  zoom_range = 0.3,
  width_shift_range = 0.3,
  height_shift_range=0.3,
  rotation_range=30
).flow_from_directory(
  VALIDATION_DATA_DIR,
  target_size = (IMG_HEIGHT, IMG_WIDTH),
  batch_size = BATCH_SIZE,
  class_mode = "categorical"
)

print('Training model...')

model.fit_generator(
  train_batches, 
  steps_per_epoch=int(N_TRAIN_SAMPLES/BATCH_SIZE) * DATA_AUGMENTATION_FACTOR,
  validation_data=validation_batches,
  validation_steps=int(N_VALIDATION_SAMPLES/BATCH_SIZE) * DATA_AUGMENTATION_FACTOR,
  epochs=EPOCHS,
  callbacks=[checkpoint, early],
  verbose=VERBOSE,
)
