from model_vgg19 import ModelVGG19

ModelVGG19(learning_rate=0.00001, n_frozen_layers=22, batch_size=128).train()