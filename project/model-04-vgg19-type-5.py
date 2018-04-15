from model_vgg19 import ModelVGG19

ModelVGG19(learning_rate=0.00001, n_freeze_layers=17, batch_size=64).train()
