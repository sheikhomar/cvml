from model_vgg19v2 import ModelVGG19v2

model = ModelVGG19v2()
path = 'saved_weights/model-06-vgg19v2-type-1-epoch82-acc0.93-loss0.20-valacc0.84-valloss0.58.hdf5'
y_test_predictions = model.predict_test(model_weights=path)

model.write_predictions(y_test_predictions, 'test-pred-model-06-vgg19v2-type-1-epoch82.csv')
