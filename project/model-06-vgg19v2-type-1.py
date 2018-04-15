from model_vgg19v2 import ModelVGG19v2

ModelVGG19v2(learning_rate=0.00001, n_freeze_layers=12, batch_size=64).train()
