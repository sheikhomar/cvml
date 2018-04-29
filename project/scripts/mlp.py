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
      "mlp__hidden_layer_sizes": [(100,), (400,), (600,), (600,), (1000,)],
      "mlp__max_iter": [5, 10, 100, 400, 700, 1000],
      'mlp__alpha': 10.0 ** -np.arange(1, 7)
    }


if __name__ == '__main__':
  MLP().run()
