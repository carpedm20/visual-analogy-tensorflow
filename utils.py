import pprint
import numpy as np

pp = pprint.PrettyPrinter()

def merge(*images):
  h, w = images[0].shape[1], images[0].shape[2]
  h_count, w_count = 4, len(images[0])
  img = np.zeros((h * h_count, w * w_count, 3))

  for idx, image_set in enumerate(zip(*images)):
    import ipdb; ipdb.set_trace() 
    for jdx, image in enumerate(image_set):
      img[jdx*h:jdx*h + h, idx*w:idx*w + w, :] = image
  return img

def imsave(path, image):
  return scipy.misc.imsave(path, image)
