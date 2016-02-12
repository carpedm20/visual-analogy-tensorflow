import pprint
import numpy as np
import scipy.misc
from time import gmtime, strftime

pp = pprint.PrettyPrinter()

strfnow = lambda: strftime("%Y-%m-%d %H:%M:%S", gmtime())

def merge(*images):
  images = list(images)
  # For difference between target and inferenced image
  images.append(abs(images[-2] - images[-1]))

  h, w = images[0].shape[1], images[0].shape[2]
  h_count, w_count = len(images), len(images[0])
  img = np.zeros((h * h_count, w * w_count, 3))

  for idx, image_set in enumerate(zip(*(images))):
    for jdx, image in enumerate(image_set):
      image[[0,-1],:,:]=1
      image[:,[0,-1],:]=1
      img[jdx*h:jdx*h + h, idx*w:idx*w + w, :] = image
  return img

def imsave(path, image):
  print(" [*] Save %s" % path)
  return scipy.misc.imsave(path, image)
