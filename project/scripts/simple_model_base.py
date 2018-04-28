import pandas as pd
import os
import json
import sys

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

class SimpleModelBase:
  def __init__(self,
               model_name=None,
               n_jobs=4,
               verbose=10,
               cv=10,
               searcher='grid'
               ):
    if model_name is None:
      script_name, script_ext = os.path.splitext(sys.argv[0])
      self.model_name = os.path.basename(script_name)
    else:
      self.model_name = model_name
    pass

    self.n_jobs = n_jobs
    self.verbose = verbose
    self.cv = cv
    self.searcher = searcher

  def run(self):
    self._load_data()
    searcher = self._get_searcher()
    searcher.fit(self.X_train_val, self.y_train_val)
    self._write_results(searcher)

  def _get_searcher(self):
    classifier = self._get_classifier()
    param_grid = self._get_search_grid_params()

    if self.searcher == 'grid':
      return GridSearchCV(
        classifier,
        param_grid,
        cv=self.cv,
        scoring='accuracy',
        verbose=self.verbose,
        n_jobs=self.n_jobs
      )
    else:
      return RandomizedSearchCV(
        classifier,
        param_grid,
        cv=self.cv,
        scoring='accuracy',
        verbose=self.verbose,
        n_jobs=self.n_jobs
      )

  def _write_results(self, searcher):
    file_path = 'simple_model_results/%s.json' % self.model_name
    print("Saving results to %s..." % file_path)
    df = pd.DataFrame(searcher.cv_results_)
    output = {
      'best_score': searcher.best_score_,
      'best_params': searcher.best_params_,
      'best_index': int(searcher.best_index_),
      'results': json.loads(df.to_json())
    }
    print(output)
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
      os.mkdir(dir_path)
    with open(file_path, 'w') as outfile:
      json.dump(output, outfile, sort_keys=True, indent=4, separators=(',', ': '))

  def _load_data(self):
    self.X_train = pd.read_csv('Train/trainVectors.csv', header=None).transpose()
    self.y_train = pd.read_csv('Train/trainLbls.csv', header=None, names=['label'])

    self.X_validation = pd.read_csv('Validation/valVectors.csv', header=None).transpose()
    self.y_validation = pd.read_csv('Validation/valLbls.csv', header=None, names=['label'])

    # X_test = pd.read_csv('Test/testVectors.csv', header=None).transpose()

    self.X_train_val = pd.concat([self.X_train, self.X_validation]).reset_index(drop=True)
    self.y_train_val = pd.concat([self.y_train, self.y_validation]).reset_index(drop=True)['label']

  def _get_classifier(self):
    raise NotImplementedError('subclasses must override _get_classifier()')

  def _get_search_grid_params(self):
    raise NotImplementedError('subclasses must override _get_search_grid_params()')
