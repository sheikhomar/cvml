from model_vgg19 import ModelVGG19

ModelVGG19(
  learning_rate=0.001,
  n_frozen_layers=7,
  batch_size=64
).train()
