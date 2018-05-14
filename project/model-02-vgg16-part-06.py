from model_vgg16 import ModelVGG16
from keras.optimizers import SGD

ModelVGG16(
  learning_rate=None,
  optimizer=SGD(momentum=0.9),
  n_freeze_layers=0,
  batch_size=64
).train()
