from model_vgg19 import ModelVGG19
from keras.optimizers import SGD

# Same as type-4, only difference is the different optimizer
ModelVGG19(
  learning_rate=None,
  optimizer=SGD(momentum=0.9),
  n_freeze_layers=12,
  batch_size=64
).train()
