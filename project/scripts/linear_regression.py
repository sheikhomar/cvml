import numpy as np
from sklearn.linear_model import LinearRegression
from simple_model_base import SimpleModelBase


class LR(SimpleModelBase):
  def __init__(self, *args, **kwargs):
    SimpleModelBase.__init__(self, *args, **kwargs)

  def _get_classifier(self):
    return LinearRegression()

  def _get_search_grid_params(self):
    return {
      'fit_intercept': [True, False],
      'normalize': [True, False]
    }


if __name__ == '__main__':
  LR().run()
