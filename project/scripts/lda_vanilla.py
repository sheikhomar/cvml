import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from simple_model_base import SimpleModelBase


class LDA(SimpleModelBase):
  def __init__(self, *args, **kwargs):
    SimpleModelBase.__init__(self, *args, **kwargs)

  def _get_classifier(self):
    return LinearDiscriminantAnalysis(solver='svd')

  def _get_search_grid_params(self):
    return {}


if __name__ == '__main__':
  LDA().predict({

  })
