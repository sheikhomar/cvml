import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

from sklearn.metrics import classification_report

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping



y_train = pd.read_csv('Train/trainLbls.csv', header=None, names=['label'])['label']
y_validation = pd.read_csv('Validation/valLbls.csv', header=None, names=['label'])['label']
X_test = pd.read_csv('Test/testVectors.csv', header=None).transpose()




img_width, img_height = 256, 256
train_data_dir = "Train/TrainImages"
validation_data_dir = "Validation/ValidationImages"
test_data_dir = "Test/TestImages"
n_train_samples = len(y_train)
n_validation_samples = len(y_validation)
n_test_samples = X_test.shape[0]
n_labels = len(y_train.unique())
batch_size = 16
epochs = 50



# ImageDataGenerator generates batches of normalised image data i.e.
# a format that the images must be in to be read by the Keras model
train_batches = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=30
).flow_from_directory(
    train_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = "categorical")
    
    
validation_batches = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=30
).flow_from_directory(
    validation_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = "categorical")


# Create a very simple CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_width, img_height, 3)),
    Flatten(),
    Dense(n_labels, activation='softmax')
])

# Compile the model 
model.compile(optimizers.Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# Save the model according to the conditions  
checkpoint = ModelCheckpoint("model-01-simple-cnn.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# Train the model
model.fit_generator(train_batches, 
                    steps_per_epoch=int(n_train_samples/batch_size),
                    validation_data=validation_batches,
                    validation_steps=int(n_validation_samples/batch_size)
                    epochs=epochs,
                    verbose=1,
                    callbacks = [checkpoint, early]
                   )
