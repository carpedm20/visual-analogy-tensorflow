import os
import scipy.io
import scipy.misc
import numpy as np
from time import gmtime, strftime
from numpy.random import choice

class Loader(object):

  def __init__(self, dataset, batch_size):
    self.dataset = dataset
    self.batch_size = batch_size
    self.options = ['rotate', 'scale', 'xpos', 'ypos']

    if dataset == "shape":
      mat_fname = "shapes48.mat"
    elif dataset == "sprite":
      mat_fname = "shapes48.mat"
    else:
      raise Exception(" [!] No dataset exists for %s." % dataset)

    mat_path = os.path.join("data", mat_fname)
    print (" [*] loading %s" % mat_path)
    mat = scipy.io.loadmat(mat_path)

    if dataset == "shape":
      self.data = mat['M']
      self.data_shape = self.data.shape
      self.data = self.data.reshape(list(self.data.shape[:3]) + [-1])

      self.width, self.height, self.channel, self.color, \
          self.shape, self.scale, self.angle, self.xpos, self.ypos = self.data_shape

      num_id = self.color * self.shape
      pair_matrix = np.eye(num_id).flatten()

      num_train = 800
      num_test = 224
      
      random_idx = choice(range(pair_matrix.size), num_train, replace=False)
      pair_matrix[random_idx] = 1

      pair_matrix = pair_matrix.reshape([num_id, num_id])
      self.train_pairs = np.array(zip(*np.nonzero(pair_matrix)))
      self.test_pairs = np.array(zip(*(pair_matrix == 0)))

      self.tests = {}
      for option in self.options:
        test_a, test_b, test_c, test_d = self.next_test(set_option=option)
        self.tests[option] = [test_a, test_b, test_c, test_d]

    elif dataset == "sprites":
      pass

  def next(self, set_option=None):
    return self.get_set_from_pairs(self.train_pairs, set_option)

  def next_test(self, set_option=None):
    return self.get_set_from_pairs(self.test_pairs, set_option)

  def get_set_from_pairs(self, pairs, set_option):
    idxes = choice(range(len(pairs)), self.batch_size)

    cur_pairs = pairs[idxes]
    cur_pairs_idx1 = cur_pairs[:,0]
    cur_pairs_idx2 = cur_pairs[:,1]

    default_angle1 = choice(self.angle, self.batch_size)
    default_scale1 = choice(self.scale, self.batch_size)
    default_xpos1 = choice(self.xpos, self.batch_size)
    default_ypos1 = choice(self.ypos, self.batch_size)

    default_angle2 = choice(self.angle, self.batch_size)
    default_scale2 = choice(self.scale, self.batch_size)
    default_xpos2 = choice(self.xpos, self.batch_size)
    default_ypos2 = choice(self.ypos, self.batch_size)

    angle1 = default_angle1
    angle2 = default_angle1
    angle3 = default_angle2
    angle4 = default_angle2
    scale1 = default_scale1
    scale2 = default_scale1
    scale3 = default_scale2
    scale4 = default_scale2

    xpos1 = default_xpos1
    xpos2 = default_xpos1
    xpos3 = default_xpos2
    xpos4 = default_xpos2
    ypos1 = default_ypos1
    ypos2 = default_ypos1
    ypos3 = default_ypos2
    ypos4 = default_ypos2

    if set_option != None:
      to_change = set_option
    else:
      to_change = choice(self.options)

    if to_change == "rotate":
      offset = choice(range(-2, 3), self.batch_size)

      angle1 = choice(self.angle, self.batch_size)
      angle2 = angle1 + offset
      angle2[angle2 < 0] += self.angle
      angle2[angle2 >= self.angle] -= self.angle

      angle3 = choice(range(self.angle), self.batch_size)
      angle4 = angle3 + offset
      angle4[angle4 < 0] += self.angle
      angle4[angle4 >= self.angle] -= self.angle
    elif to_change == "scale":
      offset = choice(range(-1, 2), self.batch_size)

      scale1 = choice(self.scale, self.batch_size)
      scale2 = scale1 + offset

      bound_idx = np.logical_or(scale2 < 0, scale2 >= self.scale)
      offset[bound_idx] *= -1
      scale2[bound_idx] = scale1[bound_idx] + offset[bound_idx]

      scale3 = choice(range(self.scale), self.batch_size)
      under_idx = np.logical_and(scale3 == 0, offset == -1)
      upper_idx = np.logical_and(scale3 == self.scale - 1, offset == 1) 
      scale3[under_idx] = choice(range(1, self.scale), np.sum(under_idx))
      scale3[upper_idx] = choice(range(0, self.scale - 1), np.sum(upper_idx))
      scale4 = scale3 + offset
    elif to_change == "xpos":
      offset = choice(range(-1, 2), self.batch_size)

      xpos1 = choice(self.xpos, self.batch_size)
      xpos2 = xpos1 + offset

      bound_idx = np.logical_or(xpos2 < 0, xpos2 >= self.xpos)
      offset[bound_idx] *= -1
      xpos2[bound_idx] = xpos1[bound_idx] + offset[bound_idx]

      xpos3 = choice(range(self.xpos), self.batch_size)
      under_idx = np.logical_and(xpos3 == 0, offset == -1)
      upper_idx = np.logical_and(xpos3 == self.xpos - 1, offset == 1) 
      xpos3[under_idx] = choice(range(1, self.xpos), np.sum(under_idx))
      xpos3[upper_idx] = choice(range(0, self.xpos - 1), np.sum(upper_idx))
      xpos4 = xpos3 + offset
    elif to_change == "ypos":
      offset = choice(range(-1, 2), self.batch_size)

      ypos1 = choice(self.ypos, self.batch_size)
      ypos2 = ypos1 + offset

      bound_idx = np.logical_or(ypos2 < 0, ypos2 >= self.ypos)
      offset[bound_idx] *= -1
      ypos2[bound_idx] = ypos1[bound_idx] + offset[bound_idx]

      ypos3 = choice(range(self.ypos), self.batch_size)
      under_idx = np.logical_and(ypos3 == 0, offset == -1)
      upper_idx = np.logical_and(ypos3 == self.ypos - 1, offset == 1) 
      ypos3[under_idx] = choice(range(1, self.ypos), np.sum(under_idx))
      ypos3[upper_idx] = choice(range(0, self.ypos - 1), np.sum(upper_idx))
      ypos4 = ypos3 + offset
    else:
      raise Exception(" [!] Wrong option %s" % to_change)
    
    color1, shape1 = np.unravel_index(cur_pairs_idx1, [self.color, self.shape])
    color2, shape2 = np.unravel_index(cur_pairs_idx2, [self.color, self.shape])

    shape = self.data_shape[3:]
    idx1 =  np.ravel_multi_index([color1, shape1, scale1, angle1, xpos1, ypos1], shape)
    idx2 =  np.ravel_multi_index([color1, shape1, scale2, angle2, xpos2, ypos2], shape)
    idx3 =  np.ravel_multi_index([color2, shape2, scale3, angle3, xpos3, ypos3], shape)
    idx4 =  np.ravel_multi_index([color2, shape2, scale4, angle4, xpos4, ypos4], shape)

    a = np.rollaxis(self.data[:,:,:,idx1], 3)
    b = np.rollaxis(self.data[:,:,:,idx2], 3)
    c = np.rollaxis(self.data[:,:,:,idx3], 3)
    d = np.rollaxis(self.data[:,:,:,idx4], 3)

    if False: # only sued for debugging
      t = strftime("%Y-%m-%d %H:%M:%S", gmtime())
      self._get_image(a, "test/%s_1.png" % t)
      self._get_image(b, "test/%s_2.png" % t)
      self._get_image(c, "test/%s_3.png" % t)
      self._get_image(d, "test/%s_4.png" % t)

    return a, b, c, d

  def _get_image(self, imgs, fname):
    for idx, img in enumerate(imgs):
      scipy.misc.imsave(fname.replace(".", "_%s." % idx).replace(" ", "_"), img)
