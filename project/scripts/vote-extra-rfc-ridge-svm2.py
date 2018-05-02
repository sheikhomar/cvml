import numpy as np

from simple_model_base import SimpleModelBase

from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier

class Voter(SimpleModelBase):
  def __init__(self, *args, **kwargs):
    SimpleModelBase.__init__(self, *args, **kwargs)

  def _get_classifier(self):
    # Test Score: 0.73641
    rfc = RandomForestClassifier(
        criterion="gini",
        max_depth=25,
        max_features="sqrt",
        n_estimators=400
    )
    
    # Test Score: 0.73352
    etc = ExtraTreesClassifier(
      max_depth=None,
      max_features=0.3,
      n_estimators=200
    )
    
    # Test Score: 0.77861
    svm = SVC(C=10, gamma=0.0001, kernel="rbf")
    
    # Test Score: 0.76878
    ridge = RidgeClassifier(alpha=580) 
    
    # Test Score: 0.76184
    ridge_bag = BaggingClassifier(
      RidgeClassifier(alpha=440),
      max_samples=0.8
    )

    return VotingClassifier(
      n_jobs=-1,
      voting='hard',
      estimators=[
        ('rfc', rfc), 
        ('etc', etc), 
        ('svm', svm), 
        ('ridge', ridge),
        ('ridge_bag', ridge_bag),
      ]
    )

  def _get_search_grid_params(self):
    return {}



if __name__ == '__main__':
  Voter().predict({
    
  })
