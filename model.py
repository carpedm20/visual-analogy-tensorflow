import tensorflow as tf

class MLP(object):
  def __init__(self, input_size=48, model_type="deep",
               n_step=200000, alpha=0.01, batch_size=25):
    self.input_size = input_size
    self.model_type = model_type
    self.n_step = n_step
    self.alpha = alpha
    self.batch_Size = batch_size

  def build_model(self):
    self.a = tf.placeholder(tf.float32, [None, self.input_size])
    self.b = tf.placeholder(tf.float32, [None, self.input_size])
    self.c = tf.placeholder(tf.float32, [None, self.input_size])
    self.d = tf.placeholder(tf.float32, [None, self.input_size])

    enc_w1 = tf.get_variable("enc_w1", [self.input_size, 4096])
    enc_w2 = tf.get_variable("enc_w2", [4096, 1024])
    enc_w3 = tf.get_variable("enc_w3", [1024, 512])

    enc_b1 = tf.get_variable("enc_b1", [4096])
    enc_b2 = tf.get_variable("enc_b2", [1024])
    enc_b3 = tf.get_variable("enc_b3", [512])

    f = tf.nn.relu
    m = tf.matmul

    enc_a = m(f(m(f(m(self.a, enc_w1) + enc_b1), enc_w2) + enc_b2), enc_w3) + enc_b3
    enc_b = m(f(m(f(m(self.b, enc_w1) + enc_b1), enc_w2) + enc_b2), enc_w3) + enc_b3
    enc_c = m(f(m(f(m(self.c, enc_w1) + enc_b1), enc_w2) + enc_b2), enc_w3) + enc_b3
    enc_d = m(f(m(f(m(self.d, enc_w1) + enc_b1), enc_w2) + enc_b2), enc_w3) + enc_b3

    if self.model_type == "add":
      self.T = (self.enc_b - self.enc_a) + self.enc_c
    elif self.model_type == "mul":
      pass
    elif self.model_type == "deep":
      self.T = (self.enc_b - self.enc_a) + self.enc_c
    else:
      raise Exception(" [!] Wrong model type : %s" % self.model_type)

    dec_w1 = tf.get_variable("dec_w1", [512, 512])
    dec_w2 = tf.get_variable("dec_w1", [512, 256])
    dec_w3 = tf.get_variable("dec_w1", [256, 512])

    dec_b1 = tf.get_variable("dec_b1", [512])
    dec_b2 = tf.get_variable("dec_b1", [256])
    dec_b3 = tf.get_variable("dec_b1", [512])

    dec_input = tf.concat(1, [T, self.enc_c])
    f(m(f(m(f(m(dec_input, dec_w1) + dec_b1), dec_w2) + dec_b2), dec_w3) + dec_b3)

    self.dec_h1 = linear(, [256])
    self.dec_h1 = linear(input_, [1024])
    self.dec_h2 = tf.nn.relu(linear(self.h1, [4096]))
    self.decoder = tf.nn.relu(linear(self.h2, [self.input_size * self.input_size]))
