import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline
from simple_model_base import SimpleModelBase


class LDA(SimpleModelBase):
  def __init__(self, *args, **kwargs):
    SimpleModelBase.__init__(self, *args, **kwargs)

  def _get_classifier(self):
    return Pipeline([
      ('preprocessing', QuantileTransformer(output_distribution='normal')),
      ('lda', LinearDiscriminantAnalysis(solver='lsqr'))
    ])

  def _get_search_grid_params(self):
    return {
      "lda__shrinkage": ['auto', 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }


if __name__ == '__main__':
  LDA().run()
