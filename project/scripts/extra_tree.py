import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from simple_model_base import SimpleModelBase


class ExtraTrees(SimpleModelBase):
  def __init__(self, *args, **kwargs):
    SimpleModelBase.__init__(self, *args, **kwargs)

  def _get_classifier(self):
    return ExtraTreesClassifier()

  def _get_search_grid_params(self):
    return {
      "n_estimators": [50, 100, 200, 300, 400, 600],
      "max_depth": [5, 10, 15, 20, 25, 30, None],
      'max_features': [0.3, 0.5, 0.8, 1]
    }


if __name__ == '__main__':
  ExtraTrees().run()
