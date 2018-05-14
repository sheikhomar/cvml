from model_base import ModelBase
from model_vgg19 import ModelVGG19

m = ModelVGG19()
pred_test = m.predict_test()
ModelBase.write_predictions(pred_test, 'test-pred-model-04-vgg19-type-1')

pred_instance = m.predict_instance_test()
ModelBase.write_predictions(pred_instance, 'test-pred-model-04-vgg19-type-1-instance')