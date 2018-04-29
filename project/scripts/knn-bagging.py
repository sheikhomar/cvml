import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

from simple_model_base import SimpleModelBase


class BaggedKNN(SimpleModelBase):
  def __init__(self, *args, **kwargs):
    SimpleModelBase.__init__(self, *args, **kwargs)

  def _get_classifier(self):
    return BaggingClassifier(KNeighborsClassifier())

  def _get_search_grid_params(self):
    return {
      'base_estimator__n_neighbors': np.arange(20, 33),
      'max_samples': [0.05, 0.1, 0.2, 0.5]
    }


if __name__ == '__main__':
  BaggedKNN().run()
