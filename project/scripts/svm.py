
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from simple_model_base import SimpleModelBase

class SVM(SimpleModelBase):
  def __init__(self, *args, **kwargs):
    SimpleModelBase.__init__(self, *args, **kwargs)

  def _get_classifier(self):
    return Pipeline([
      ('scaler', StandardScaler()),
      ('svm', SVC(kernel='rbf'))
    ])

  def _get_search_grid_params(self):
    return {
      'svm__C': [0.01, 0.1, 1, 10, 100, 1000],
      'svm__gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001]
    }


if __name__ == '__main__':
    SVM().run()
