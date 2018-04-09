import os
import re

def find_best_model(model_name, models_dir='./models/'):
  list_of_files = sorted(os.listdir(models_dir))
  best_model = None
  best_acc = 0
  best_loss = float('inf')
  for f in list_of_files:
    if f.startswith(model_name):
      values = re.findall('val[^\d]*(\d+\.\d*)', f)
      acc = float(values[0])
      loss = float(values[1])
      if acc > best_acc and loss < best_loss:
          best_acc = acc
          best_loss = loss
          best_model = os.path.join(models_dir, f)
  return best_model

