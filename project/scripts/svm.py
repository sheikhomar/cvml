import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV

def svm():
  print('Loading data...')
  X_train = pd.read_csv('Train/trainVectors.csv', header=None).transpose()
  y_train = pd.read_csv('Train/trainLbls.csv', header=None, names=['label'])

  X_validation = pd.read_csv('Validation/valVectors.csv', header=None).transpose()
  y_validation = pd.read_csv('Validation/valLbls.csv', header=None, names=['label'])

  #X_test = pd.read_csv('Test/testVectors.csv', header=None).transpose()

  X_train_val = pd.concat([X_train, X_validation]).reset_index(drop=True)
  y_train_val = pd.concat([y_train, y_validation]).reset_index(drop=True)['label']


  classifier = SVC()

  print(classifier)

  param_grid = [
    {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
  ]

  gs = GridSearchCV(classifier, param_grid, cv=10, scoring='accuracy', verbose=10, n_jobs=4)
  model = gs.fit(X_train_val, y_train_val)
  
  print("The best classifier is: ", gs.best_estimator_)

  print(gs.grid_scores_)
  
  print(model)

  print('Best score: {}. Best parameters: {}'.format(model.best_score_, model.best_params_))

if __name__ == '__main__':
    svm()
