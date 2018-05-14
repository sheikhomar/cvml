from model_base import ModelBase
from model_vgg19 import ModelVGG19

m = ModelVGG19()

weights_path = 'saved_weights/model-04-vgg19-type-1-epoch46-acc0.9164-loss0.2398-valacc0.8824-valloss0.4505.hdf5'

pred_test = m.predict_test(weights_path)
ModelBase.write_predictions(pred_test, 'test-pred-model-04-vgg19-type-1')

pred_instance = m.predict_instance_test(weights_path)
ModelBase.write_predictions(pred_instance, 'test-pred-model-04-vgg19-type-1-instance')