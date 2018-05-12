import numpy as np
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from simple_model_base import SimpleModelBase


class KNN(SimpleModelBase):
  def __init__(self, *args, **kwargs):
    SimpleModelBase.__init__(self, *args, **kwargs)

  def _get_classifier(self):
    return Pipeline([
      ('pca', PCA()),
      ('knn', KNeighborsClassifier())
    ])

  def _get_search_grid_params(self):
    return {
      'pca__n_components': np.arange(10, 100, 10),
      'knn__n_neighbors': np.arange(2, 20),
      'knn__metric': ['manhattan', 'euclidean']
    }


if __name__ == '__main__':
  KNN(n_jobs=1).predict({
    'knn__metric': 'manhattan',
    'knn__n_neighbors': 18,
    'pca__n_components': 70
  })
