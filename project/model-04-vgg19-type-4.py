from model_vgg19 import ModelVGG19

# Same as type-3, only difference is the larger batch size
ModelVGG19(
  learning_rate=0.00001,
  n_freeze_layers=12,
  batch_size=64
).train()
