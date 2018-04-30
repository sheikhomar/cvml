import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from simple_model_base import SimpleModelBase


class MLP(SimpleModelBase):
  def __init__(self, *args, **kwargs):
    SimpleModelBase.__init__(self, *args, **kwargs)

  def _get_classifier(self):
    return Pipeline([
      ('scaler', StandardScaler()),
      ('mlp', MLPClassifier())
    ])

  def _get_search_grid_params(self):
    return {
      "mlp__hidden_layer_sizes": [(30,), (50,), (100,), (150,), (200,), (10,10), (20,10),],
      "mlp__max_iter": [3, 5, 10, 15, 20, 50],
      'mlp__alpha': [1, 5, 10, 15, 20, 30, 50, 100]
    }


if __name__ == '__main__':
  MLP().run()
