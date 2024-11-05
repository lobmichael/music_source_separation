import numpy as np
import os

def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            print('Could not create directory:' + path)

def list_to_array(m):
  M = m[0]
  for i in range(1,len(m)):
    M = np.concatenate((M,m[i]), axis = 0)

  return M

def pickle(array, path, filename):
  create_dir(path)
  np.save(path+filename, array)
  return 0
