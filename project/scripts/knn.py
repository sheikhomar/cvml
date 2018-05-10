import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from simple_model_base import SimpleModelBase


class KNN(SimpleModelBase):
  def __init__(self, *args, **kwargs):
    SimpleModelBase.__init__(self, *args, **kwargs)

  def _get_classifier(self):
    # Estimate a covariance matrix
    covariance_matrix = np.cov(self._get_train_data())

    # Create a pipeline
    return Pipeline([
      ('scaler', StandardScaler()),
      ('knn',
        KNeighborsClassifier(
          metric_params={'V': covariance_matrix}
        )
      )
    ])

  def _get_search_grid_params(self):
    return {
      'knn__n_neighbors': np.arange(3, 29),
      'knn__metric': ['manhatten', 'euclidean', 'mahalanobis']
    }


if __name__ == '__main__':
  KNN(n_jobs=-1).run()
