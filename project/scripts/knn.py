import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from simple_model_base import SimpleModelBase


class KNN(SimpleModelBase):
  def __init__(self, *args, **kwargs):
    SimpleModelBase.__init__(self, *args, **kwargs)

  def _get_classifier(self):
    return Pipeline([
      ('scaler', StandardScaler()),
      ('knn', KNeighborsClassifier())
    ])

  def _get_search_grid_params(self):
    return {
      'knn__n_neighbors': np.arange(3, 29),
      'knn__metric': ['manhattan', 'euclidean']
    }


if __name__ == '__main__':
  KNN(n_jobs=-1).run()
