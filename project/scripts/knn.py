import numpy as np

from sklearn.ensemble import KNeighborsClassifier
from simple_model_base import SimpleModelBase


class KNN(SimpleModelBase):
  def __init__(self, *args, **kwargs):
    SimpleModelBase.__init__(self, *args, **kwargs)

  def _get_classifier(self):
    return KNeighborsClassifier()

  def _get_search_grid_params(self):
    return {
      'n_neighbors': np.arange(3, 33)
    }


if __name__ == '__main__':
  KNN().run()