import os
import scipy.io
import numpy as np
from numpy.random import choice, randint

class Loader(object):

  def __init__(self, dataset, batch_size):
    self.dataset = dataset
    self.batch_size = batch_size

    if dataset == "shapes":
      mat_fname = "shapes48.mat"
    elif dataset == "sprites":
      mat_fname = "shapes48.mat"
    else:
      raise Exception(" [!] No dataset exists for %s." % dataset)

    mat_path = os.path.join("data", mat_fname)
    mat = scipy.io.loadmat(mat_path)

    if dataset == "shapes":
      mat_fname = "shapes48.mat"

      data = mat['M']
      self.width, self.height, self.channel, self.color, \
          self.shape, self.scale, self.angle, self.row, self.col = data.shape

      num_id = self.color * self.shape
      pair_matrix = np.eye(num_id).flatten()

      num_train = 800
      num_test = 224
      
      random_idx = choice(range(pair_matrix.size), num_train, replace=False)
      pair_matrix[random_idx] = 1

      pair_matrix = pair_matrix.reshape([num_id, num_id])
      self.pairs = zip(*np.nonzero(pair_matrix))

    elif dataset == "sprites":
      mat_fname = "shapes48.mat"

  def next(self):
    idxes = choice(range(len(self.pairs)), 25)
    cur_pairs = [self.pairs[idx] for idx in idxes]

    angle1 = choice(self.angle, self.batch_size)
    scale1 = choice(self.scale, self.batch_size)
    xpos1 = choice(self.row, self.batch_size)
    ypos1 = choice(self.col, self.batch_size)

    angle2 = choice(self.angle, self.batch_size)
    scale2 = choice(self.scale, self.batch_size)
    xpos2 = choice(self.row, self.batch_size)
    ypos2 = choice(self.col, self.batch_size)

    to_change = randint(4)

    import ipdb; ipdb.set_trace() 
    if to_change == 0: # change angle
      offset = choice(range(-2, 3), self.batch_size)
      angle1 = choice(self.angle, self.batch_size)
      angle2 = angle1 + offset
