import time
import tensorflow as tf

from .ops import conv2d
from .base import Model
from loader import Loader

class SpriteAnalogy(Model):
  """Deep Visual Analogy Network."""
  def __init__(self, sess, image_size=48, num_hid=512,
               model_type="dis+cls", batch_size=25, dataset="shape"):
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
    self.num_hid = num_hid
    #self.loader = Loader(self.dataset, self.batch_size)

    self.cards = [2, 4, 3, 6, 2, 2, 2]
    num_categorical = 0
    for card in self.cards:
        num_categorical = num_categorical + card
    self.cards[6] = 3;
    num_categorical = num_categorical + 1
    self.num_categorical = num_categorical

    self.id_idxes = range(0, self.num_categorical - 1)
    self.pose_idxes = range(self.num_categorical, self.num_hid)

    # parameters used to save a checkpoint
    self._attrs = ['max_iter', 'batch_size', 'alpha', 'learning_rate']

    self.build_model()

  def build_model(self):
    self.a = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, 3])
    self.b = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, 3])
    self.c = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, 3])
    self.d = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, 3])

    f = tf.nn.relu
    m = tf.matmul

    with tf.variable_scope("encoder") as scope:
      enc_w1 = tf.get_variable("enc_w1", [4608, 2048])
      enc_w2 = tf.get_variable("enc_w2", [2048, 512])

      enc_b1 = tf.get_variable("enc_b1", [2048])
      enc_b2 = tf.get_variable("enc_b2", [512])

      conv_a = tf.reshape(f(conv2d(f(conv2d(self.a, 64, name="conv1")), 32, name="conv2")), [self.batch_size, -1])
      f_a = (m(f(m(conv_a, enc_w1) + enc_b1), enc_w2) + enc_b2)

      scope.reuse_variables()

      conv_b = tf.reshape(f(conv2d(f(conv2d(self.b, 64, name="conv1")), 32, name="conv2")), [self.batch_size, -1])
      f_b = (m(f(m(conv_b, enc_w1) + enc_b1), enc_w2) + enc_b2)
      conv_c = tf.reshape(f(conv2d(f(conv2d(self.c, 64, name="conv1")), 32, name="conv2")), [self.batch_size, -1])
      f_c = (m(f(m(conv_c, enc_w1) + enc_b1), enc_w2) + enc_b2)
      conv_d = tf.reshape(f(conv2d(f(conv2d(self.d, 64, name="conv1")), 32, name="conv2")), [self.batch_size, -1])
      f_d = (m(f(m(conv_d, enc_w1) + enc_b1), enc_w2) + enc_b2)

    # Transform
    T = f_b - f_a
    T = T[self.pose_idxes, :]

    top_pose = f_c[self.pose_idxes, :]
    sw = np.ones_like(T)

    g_input = T + f_c

    dec_w1 = tf.get_variable("dec_w1", [g_input.get_shape()[-1], 1024])
    dec_w2 = tf.get_variable("dec_w2", [1024, 4096])
    dec_w3 = tf.get_variable("dec_w3", [4096, self.image_size * self.image_size * 3])

    dec_b1 = tf.get_variable("dec_b1", [1024])
    dec_b2 = tf.get_variable("dec_b2", [4096])
    dec_b3 = tf.get_variable("dec_b3", [self.image_size * self.image_size * 3])

    self.g = f(m(f(m(f(m(T, dec_w1) + dec_b1), dec_w2) + dec_b2), dec_w3) + dec_b3)

    self.g_img = tf.reshape(self.g, [self.batch_size, self.image_size, self.image_size, 3])
    _ = tf.image_summary("g", self.g_img, max_images=5)

    self.l = tf.nn.l2_loss(d - self.g)
    _ = tf.scalar_summary("loss", self.l)

    self.r = tf.nn.l2_loss(f_d - f_c - T)
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

    self.loss = (self.l + self.alpha * self.r) / self.batch_size
    _ = tf.scalar_summary("l_plus_r", self.loss)

    self.lr = tf.train.exponential_decay(self.learning_rate,
                                         global_step=self.step,
                                         decay_steps=100000,
                                         decay_rate=0.999)
    self.optim = tf.train.MomentumOptimizer(self.lr, momentum=0.9) \
                         .minimize(self.loss, global_step=self.step)

    merged_sum = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./logs", self.sess.graph_def)

    tf.initialize_all_variables().run()

    start_time = time.time()
    for step in xrange(self.max_iter):
      if step % 1000  == 0:
        self.save(checkpoint_dir)

      if step % 2  == 0:
        feed = {self.a: self.loader.test_a,
                self.b: self.loader.test_b,
                self.c: self.loader.test_c,
                self.d: self.loader.test_d}

        summary_str, loss = self.sess.run([merged_sum, self.loss], feed_dict=feed)
        writer.add_summary(summary_str, step)

        if step % 50 == 0:
          print("Epoch: [%2d/%7d] time: %4.4f, loss: %.8f" % (step, self.max_iter, time.time() - start_time, loss))

      a, b, c, d = self.loader.next()

      feed = {self.a: a,
              self.b: b,
              self.c: c,
              self.d: d}
      self.sess.run(self.optim, feed_dict=feed)
