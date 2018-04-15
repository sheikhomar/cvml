from model_inception_v3 import ModelInceptionV3

ModelInceptionV3(learning_rate=0.00001, n_freeze_layers=0, batch_size=8).train()
