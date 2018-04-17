from model_inception_v3_take_2 import ModelInceptionV3Take2

ModelInceptionV3Take2(
    learning_rate=0.0000001,
    n_freeze_layers=0,
    batch_size=2,
    verbose=1
).train()
