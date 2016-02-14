import pprint
import numpy as np
import scipy.misc
from time import gmtime, strftime

pp = pprint.PrettyPrinter()

strfnow = lambda: strftime("%Y-%m-%d %H:%M:%S", gmtime())

def merge(*images):
  images = list(images)
  # For difference between target and inferenced image
  # images.append(abs(images[-2] - images[-1]))

  h, w = images[0].shape[1], images[0].shape[2]
  h_count, w_count = len(images), len(images[0])
  img = np.zeros((h * h_count, w * w_count, 3))

  for idx, image_set in enumerate(zip(*(images))):
    for jdx, image in enumerate(image_set):
      copy_img = image.copy()
      copy_img[[0,-1],:,:]=1
      copy_img[:,[0,-1],:]=1
      img[jdx*h:jdx*h + h, idx*w:idx*w + w, :] = copy_img
  return img

def imsave(path, image):
  print(" [*] Save %s" % path)
  image[image>1] = 1
  image[image<0] = 0
  return scipy.misc.imsave(path, image)

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)
