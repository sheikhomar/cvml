import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from simple_model_base import SimpleModelBase


class LDA(SimpleModelBase):
  def __init__(self, *args, **kwargs):
    SimpleModelBase.__init__(self, *args, **kwargs)

  def _get_classifier(self):
    return LinearDiscriminantAnalysis()

  def _get_search_grid_params(self):
    return {
      "shrinkage": ['auto', 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
      "solver": ['lsqr', 'eigen'],
      "n_components": [10, 13, 17, 19, 23, 26, 28]
    }


if __name__ == '__main__':
  LDA().run()
