import os

class Loader(object):

  def __init__(self, dataset):
    if dataset == "shapes":
      mat_fname = "shapes48.mat"
    elif dataset == "sprites":
      mat_fname = "shapes48.mat"
    else:
      raise Exception(" [!] No dataset exists for %s." % dataset)

    self.data_path = os.path.join("data", mat_fname)

