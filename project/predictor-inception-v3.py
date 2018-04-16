from model_inception_v3 import ModelInceptionV3

model = ModelInceptionV3()
path = 'saved_weights/model-03-inceptionv3-part-01-epoch39-acc0.93-loss0.26-valacc0.86-valloss0.58.hdf5'
y_test_predictions = model.predict_test(model_weights=path)

model.write_predictions(y_test_predictions, 'test-pred-model-03-inceptionv3-part-01-epoch39.csv')
