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
      ('mlp', MLPClassifier(activation='logistic', solver='lbfgs'))
    ])

  def _get_search_grid_params(self):
    return {
      'mlp__hidden_layer_sizes': [(10,), (15,), (20,), (25,), (30,), (40,), (50,), (70,)],
      'mlp__alpha': [5, 10, 15, 17, 20, 22, 24, 27]
    }


if __name__ == '__main__':
  MLP().run()
