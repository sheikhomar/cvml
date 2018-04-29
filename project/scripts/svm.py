
from sklearn.svm import SVC

from simple_model_base import SimpleModelBase

class SVM(SimpleModelBase):
  def __init__(self, *args, **kwargs):
    SimpleModelBase.__init__(self, *args, **kwargs)

  def _get_classifier(self):
    return SVC()

  def _get_search_grid_params(self):
    return {
      'C': [0.01, 0.1, 1, 10, 100, 1000],
      'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']
    }


if __name__ == '__main__':
    SVM().predict()
