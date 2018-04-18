from model_vgg19v3 import ModelVGG19v3

model = ModelVGG19v3()
path = 'saved_weights/model-08-vgg19v3-type-1-epoch78-acc0.94-loss0.16-valacc0.87-valloss0.54.hdf5'
y_test_predictions = model.predict_test(model_weights=path)

model.write_predictions(y_test_predictions, 'test-pred-model-08-vgg19v3-type-1-epoch78.csv')
