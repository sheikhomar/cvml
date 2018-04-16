from model_inception_v3_take_2 import ModelInceptionV3Take2

ModelInceptionV3Take2(
    learning_rate=0.00001,
    n_freeze_layers=311,
    batch_size=2,
    verbose=1
).train()
