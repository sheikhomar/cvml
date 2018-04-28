import numpy as np

from sklearn.ensemble import RandomForestClassifier
from simple_model_base import SimpleModelBase


class RandomForest(SimpleModelBase):
  def __init__(self, *args, **kwargs):
    SimpleModelBase.__init__(self, *args, **kwargs)

  def _get_classifier(self):
    return RandomForestClassifier()

  def _get_search_grid_params(self):
    return {
      "n_estimators": [50, 100, 200, 300, 400, 600],
      "max_depth": [5, 10, 15, 20, 25, 30, None],
      'max_features': ['sqrt', 'log2'],
      'criterion': ['gini', 'entropy']
    }


if __name__ == '__main__':
  RandomForest().run()
