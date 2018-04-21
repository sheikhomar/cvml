from model_vgg19v3 import ModelVGG19v3

ModelVGG19v3(
    learning_rate=0.0001,
    n_freeze_layers=12,
    batch_size=128
).train()
