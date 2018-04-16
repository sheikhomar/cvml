from model_vgg16v2 import ModelVGG16v2

ModelVGG16v2(
    learning_rate=0.000001,
    n_freeze_layers=0,
    batch_size=128
).train()
