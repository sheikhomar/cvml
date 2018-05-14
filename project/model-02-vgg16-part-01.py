from model_vgg16 import ModelVGG16

m = ModelVGG16(
  learning_rate=0.00001,
  n_freeze_layers=0,
  batch_size=64
)
m.train()
m.predict_test()
m.predict_instance_test()
