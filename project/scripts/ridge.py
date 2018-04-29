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
      'alpha': np.arange(100, 1000, 10)
    }


if __name__ == '__main__':
  Ridge().run()
