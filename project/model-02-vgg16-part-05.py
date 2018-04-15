from model_vgg16 import ModelVGG16

ModelVGG16(learning_rate=0.00001, n_freeze_layers=15, batch_size=64).train()
