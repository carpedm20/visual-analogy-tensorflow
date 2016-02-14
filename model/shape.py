import os
import time
import tensorflow as tf

from .base import Model
from loader import Loader
from utils import merge, imsave, strfnow

class ShapeAnalogy(Model):
  """Deep Visual Analogy Network."""
  def __init__(self, sess, image_size=48, model_type="deep",
               batch_size=25, dataset="shape"):
    """Initialize the parameters for an Deep Visual Analogy network.

    Args:
      image_size: int, The size of width and height of input image
      model_type: string, The type of increment function ["add", "deep"]
      batch_size: int, The size of a batch [25]
      dataset: str, The name of dataset ["shape", ""]
    """
    self.sess = sess

    self.image_size = image_size
    self.model_type = model_type
    self.batch_size = batch_size
    self.dataset = dataset
    self.loader = Loader(self.dataset, self.batch_size)

    self.sample_dir = "samples"
    if not os.path.exists(self.sample_dir):
      os.makedirs(self.sample_dir)

    # parameters used to save a checkpoint
    self._attrs = ['batch_size', 'model_type', 'image_size']
    self.options = ['rotate', 'scale', 'xpos', 'ypos']

    self.build_model()

  def build_model(self):
    self.a = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3])
    self.b = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3])
    self.c = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3])
    self.d = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3])

    a = tf.reshape(self.a, [self.batch_size, self.image_size * self.image_size * 3])
    b = tf.reshape(self.b, [self.batch_size, self.image_size * self.image_size * 3])
    c = tf.reshape(self.c, [self.batch_size, self.image_size * self.image_size * 3])
    d = tf.reshape(self.d, [self.batch_size, self.image_size * self.image_size * 3])

    enc_w1 = tf.get_variable("enc_w1", [self.image_size * self.image_size * 3, 4096])
    enc_w2 = tf.get_variable("enc_w2", [4096, 1024])
    enc_w3 = tf.get_variable("enc_w3", [1024, 512])

    enc_b1 = tf.get_variable("enc_b1", [4096])
    enc_b2 = tf.get_variable("enc_b2", [1024])
    enc_b3 = tf.get_variable("enc_b3", [512])

    f = tf.nn.relu
    m = tf.matmul

    f_a = m(f(m(f(m(a, enc_w1) + enc_b1), enc_w2) + enc_b2), enc_w3) + enc_b3
    f_b = m(f(m(f(m(b, enc_w1) + enc_b1), enc_w2) + enc_b2), enc_w3) + enc_b3
    f_c = m(f(m(f(m(c, enc_w1) + enc_b1), enc_w2) + enc_b2), enc_w3) + enc_b3
    f_d = m(f(m(f(m(d, enc_w1) + enc_b1), enc_w2) + enc_b2), enc_w3) + enc_b3

    if self.model_type == "add":
      T = (f_b - f_a)
    elif self.model_type == "deep":
      T_input = tf.concat(1, [f_b - f_a, f_c])

      deep_w1 = tf.get_variable("deep_w1", [1024, 512])
      deep_w2 = tf.get_variable("deep_w2", [512, 256])
      deep_w3 = tf.get_variable("deep_w3", [256, 512])

      deep_b1 = tf.get_variable("deep_b1", [512])
      deep_b2 = tf.get_variable("deep_b2", [256])
      deep_b3 = tf.get_variable("deep_b3", [512])

      T = m(f(m(f(m(T_input, deep_w1) + deep_b1), deep_w2) + deep_b2), deep_w3) + deep_b3
    else:
      raise Exception(" [!] Wrong model type : %s" % self.model_type)

    dec_w1 = tf.get_variable("dec_w1", [T.get_shape()[-1], 1024])
    dec_w2 = tf.get_variable("dec_w2", [1024, 4096])
    dec_w3 = tf.get_variable("dec_w3", [4096, self.image_size * self.image_size * 3])

    dec_b1 = tf.get_variable("dec_b1", [1024])
    dec_b2 = tf.get_variable("dec_b2", [4096])
    dec_b3 = tf.get_variable("dec_b3", [self.image_size * self.image_size * 3])

    self.g1 = m(f(m(f(m(T + f_c, dec_w1) + dec_b1), dec_w2) + dec_b2), dec_w3) + dec_b3
    self.g2 = m(f(m(f(m(2*T + f_c, dec_w1) + dec_b1), dec_w2) + dec_b2), dec_w3) + dec_b3
    self.g3 = m(f(m(f(m(3*T + f_c, dec_w1) + dec_b1), dec_w2) + dec_b2), dec_w3) + dec_b3

    self.g1_img = tf.reshape(self.g1, [self.batch_size, self.image_size, self.image_size, 3])
    self.g2_img = tf.reshape(self.g2, [self.batch_size, self.image_size, self.image_size, 3])
    self.g3_img = tf.reshape(self.g3, [self.batch_size, self.image_size, self.image_size, 3])
    _ = tf.image_summary("g", self.g1_img, max_images=5)

    self.l = tf.nn.l2_loss(d - self.g1) / self.batch_size
    _ = tf.scalar_summary("loss", self.l)

    self.r = tf.nn.l2_loss(f_d - f_c - T) / self.batch_size
    _ = tf.scalar_summary("regularizer", self.r)

  def train(self, max_iter=450000,
            alpha=0.01, learning_rate=0.001,
            checkpoint_dir="checkpoint"):
    """Train an Deep Visual Analogy network.

    Args:
      max_iter: int, The size of total iterations [450000]
      alpha: float, The importance of regularizer term [0.01]
      learning_rate: float, The learning rate of SGD [0.001]
      checkpoint_dir: str, The path for checkpoints to be saved [checkpoint]
    """
    self.max_iter = max_iter
    self.alpha = alpha
    self.learning_rate = learning_rate
    self.checkpoint_dir = checkpoint_dir

    self.step = tf.Variable(0, trainable=False)

    self.loss = (self.l + self.alpha * self.r)
    _ = tf.scalar_summary("l_plus_r", self.loss)

    self.lr = tf.train.exponential_decay(self.learning_rate,
                                         global_step=self.step,
                                         decay_steps=100000,
                                         decay_rate=0.999)
    self.optim = tf.train.MomentumOptimizer(self.lr, momentum=0.9) \
                         .minimize(self.loss, global_step=self.step)
    #self.optim = tf.train.AdamOptimizer(self.lr, beta1=0.5) \
    #                     .minimize(self.loss, global_step=self.step)
    #self.optim = tf.train.RMSPropOptimizer(self.lr, momentum=0.9, decay=0.95) \
    #                     .minimize(self.loss, global_step=self.step)

    merged_sum = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./logs", self.sess.graph_def)

    tf.initialize_all_variables().run()
    self.load(self.checkpoint_dir)

    start_time = time.time()
    start_iter = self.step.eval()

    test_a, test_b, test_c, test_d = self.loader.tests['rotate']

    for step in xrange(start_iter, start_iter + self.max_iter):
      if step != 0 and step % 10000 == 0:
        self.test(fixed=True)
        self.save(checkpoint_dir, step)

      if step % 5  == 1:
        feed = {self.a: test_a, self.b: test_b, self.c: test_c, self.d: test_d}

        summary_str, loss = self.sess.run([merged_sum, self.loss], feed_dict=feed)
        writer.add_summary(summary_str, step)

        if step % 50 == 1:
          print("Epoch: [%2d/%7d] time: %4.4f, loss: %.8f" % (step, self.max_iter, time.time() - start_time, loss))

      a, b, c, d = self.loader.next()

      feed = {self.a: a,
              self.b: b,
              self.c: c,
              self.d: d}
      self.sess.run(self.optim, feed_dict=feed)

  def test(self, name="test", options=None, fixed=False):
    if options == None:
      options = self.options

    t = strfnow()

    for option in options:
      if fixed == True:
        a, b, c, d = self.loader.tests[option]
      else:
        a, b, c, d = self.loader.next(set_option=option)

      feed = {self.a: a,
              self.b: b,
              self.c: c,
              self.d: d}

      fname = "%s/%s_option:%s_time:%s.png" % (self.sample_dir, name, option, t)
      g_img, g2_img, g3_img = self.sess.run([self.g1_img, self.g2_img, self.g3_img], feed_dict=feed)

      imsave(fname, merge(a, b, c, d, g_img, g2_img, g3_img))
