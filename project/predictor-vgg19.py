from model_vgg19 import ModelVGG19

model = ModelVGG19()
path = 'saved_weights/model-04-vgg19-type-4/model-04-vgg19-type-4-epoch72-acc0.90-loss0.29-valacc0.84-valloss0.53.hdf5'
y_test_predictions = model.predict_test(model_weights=path)

model.write_predictions(y_test_predictions, 'test-pred-model-04-vgg19-type-4-epoch72.csv')
