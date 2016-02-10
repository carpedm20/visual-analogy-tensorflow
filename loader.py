import os
import scipy.io
import numpy as np
from numpy.random import choice, randint

class Loader(object):

  def __init__(self, dataset, batch_size):
    self.dataset = dataset
    self.batch_size = batch_size

    if dataset == "shape":
      mat_fname = "shapes48.mat"
    elif dataset == "sprite":
      mat_fname = "shapes48.mat"
    else:
      raise Exception(" [!] No dataset exists for %s." % dataset)

    mat_path = os.path.join("data", mat_fname)
    print (" [*] loading %s" % mat_path)
    mat = scipy.io.loadmat(mat_path)

    if dataset == "shapes":
      mat_fname = "shapes48.mat"

      self.data = mat['M']
      self.data_shape = self.data.shape
      self.data = self.data.reshape(list(self.data.shape[:3]) + [-1])

      self.width, self.height, self.channel, self.color, \
          self.shape, self.scale, self.angle, self.row, self.col = self.data_shape

      num_id = self.color * self.shape
      pair_matrix = np.eye(num_id).flatten()

      num_train = 800
      num_test = 224
      
      random_idx = choice(range(pair_matrix.size), num_train, replace=False)
      pair_matrix[random_idx] = 1

      pair_matrix = pair_matrix.reshape([num_id, num_id])
      self.pairs = np.array(zip(*np.nonzero(pair_matrix)))

    elif dataset == "sprites":
      mat_fname = "shapes48.mat"

  def next(self):
    idxes = choice(range(len(self.pairs)), 25)

    cur_pairs = self.pairs[idxes]
    cur_pairs_idx1 = cur_pairs[:,0]
    cur_pairs_idx2 = cur_pairs[:,1]

    angle1 = choice(self.angle, self.batch_size)
    scale1 = choice(self.scale, self.batch_size)
    xpos1 = choice(self.row, self.batch_size)
    ypos1 = choice(self.col, self.batch_size)

    angle2 = choice(self.angle, self.batch_size)
    scale2 = choice(self.scale, self.batch_size)
    xpos2 = choice(self.row, self.batch_size)
    ypos2 = choice(self.col, self.batch_size)

    to_change = randint(4)

    if to_change == 0: # change angle
      offset = choice(range(-2, 3), self.batch_size)

      angle1 = choice(self.angle, self.batch_size)
      angle2 = angle1 + offset
      angle2[angle2 < 0] += self.angle
      angle2[angle2 >= self.angle] -= self.angle

      angle3 = choice(range(self.angle), self.batch_size)
      angle4 = angle3 + offset
      angle4[angle4 < 0] += self.angle
      angle4[angle4 >= self.angle] -= self.angle
    elif to_change == 1: # change scale
      offset = choice(range(-1, 2), self.batch_size)

      scale1 = choice(self.scale, self.batch_size)
      scale2 = scale1 + offset

      import ipdb; ipdb.set_trace() 
      bound_idx = scale2 < 0 | scale2 >= self.scale
      offset[bound_idx] *= 1
      scale2[bound_idx] = scale1[bound_idx] + offset[bound_idx]

      scale3 = choice(range(self.scale), self.batch_size)
      under_idx = scale3 < 0
      upper_idx = scale3 >= self.sacle
      scale3[under_idx] = choice(range(1, self.scale), len(under_idx))
      scale3[upper_idx] = choice(range(0, self.scale - 1), len(upper_idx))
      scale4 = scale3 + offset
    elif to_change == 2:
      pass
    elif to_change == 3:
      pass
    
    color1, shape1 = np.unravel_index(cur_pairs_idx1, [self.color, self.shape])
    color2, shape2 = np.unravel_index(cur_pairs_idx2, [self.color, self.shape])

    shape = self.data_shape[3:]
    idx1 =  np.ravel_multi_index([color1, shape1, scale1, angle1, xpos1, ypos1], shape)
    idx2 =  np.ravel_multi_index([color2, shape2, scale2, angle2, xpos2, ypos2], shape)
    idx3 =  np.ravel_multi_index([color3, shape3, scale3, angle3, xpos3, ypos3], shape)
    idx4 =  np.ravel_multi_index([color4, shape4, scale4, angle4, xpos4, ypos4], shape)

    a = self.data[:,:,:,idx1]
    b = self.data[:,:,:,idx2]
    c = self.data[:,:,:,idx3]
    d = self.data[:,:,:,idx4]

    return a, b, c, d
