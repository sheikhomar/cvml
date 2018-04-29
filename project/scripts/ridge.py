import numpy as np
from sklearn.linear_model import RidgeClassifier
from simple_model_base import SimpleModelBase


class Ridge(SimpleModelBase):
  def __init__(self, *args, **kwargs):
    SimpleModelBase.__init__(self, *args, **kwargs)

  def _get_classifier(self):
    return RidgeClassifier()

  def _get_search_grid_params(self):
    return {
      'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 1, 3, 5, 10]
    }


if __name__ == '__main__':
  Ridge().run()
