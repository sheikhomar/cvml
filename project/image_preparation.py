import pandas as pd
import os


def organise_images(labels, dir_name='Train/TrainImages'):
  for i in range(0, len(labels)):
    old_name = '%s/Image%s.jpg' % (dir_name, i + 1)
    if os.path.exists(old_name):
      new_name = '%s/%s/Image%s.jpg' % (dir_name, labels[i], i + 1)
      new_dir = os.path.dirname(new_name)
      if not os.path.exists(new_dir):
        os.makedirs(new_dir)
      os.rename(old_name, new_name)
      # print('.', end='')
      # print('{} -> {}'.format(old_name, new_name))
    else:
      print('!', end='')


y_train = pd.read_csv('Train/trainLbls.csv', header=None, names=['label'])['label']
y_validation = pd.read_csv('Validation/valLbls.csv', header=None, names=['label'])['label']

organise_images(y_train, 'Train/TrainImages')
organise_images(y_validation, 'Validation/ValidationImages')