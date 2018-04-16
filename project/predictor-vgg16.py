from model_vgg16 import ModelVGG16

model = ModelVGG16()
path = 'saved_weights/model-02-vgg16-part-01/model-02-vgg16-part-01-epoch64-acc0.90-loss0.29-valacc0.85-valloss0.51.hdf5'
y_test_predictions = model.predict_test(model_weights=path)

model.write_predictions(y_test_predictions, 'test-pred-model-02-vgg16-part-01-epoch64.csv')
