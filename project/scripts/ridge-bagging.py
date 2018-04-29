import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import BaggingClassifier
from simple_model_base import SimpleModelBase


class RidgeBagging(SimpleModelBase):
  def __init__(self, *args, **kwargs):
    SimpleModelBase.__init__(self, *args, **kwargs)

  def _get_classifier(self):
    return BaggingClassifier(RidgeClassifier())

  def _get_search_grid_params(self):
    return {
      'base_estimator__alpha': np.arange(10, 500, 10),
      'max_samples': [0.3, 0.5, 0.7, 0.8, 1.0]
    }


if __name__ == '__main__':
  RidgeBagging().run()
