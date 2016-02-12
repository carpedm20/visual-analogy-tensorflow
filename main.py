import os
import tensorflow as tf

from model import ShapeAnalogy, SpriteAnalogy
from utils import pp

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.0001]")
flags.DEFINE_float("alpha", 0.001, "The importance of regularizer term [0.01]")
flags.DEFINE_integer("max_iter", 450000, "The size of total iterations [450000]")
flags.DEFINE_integer("batch_size", 25, "The size of batch images [25]")
flags.DEFINE_integer("image_size", 48, "The size of width or height of image to use [48]")
flags.DEFINE_string("dataset", "shape", "The name of dataset [shape, sprite]")
flags.DEFINE_string("model_type", "deep", "The type of the model [add, deep]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

model_dict = {
  "shape": ShapeAnalogy,
  "sprite": SpriteAnalogy
}

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

  Analogy = model_dict[FLAGS.dataset]

  with tf.Session() as sess:
    analogy = Analogy(sess, image_size=FLAGS.image_size, model_type=FLAGS.model_type,
                      batch_size=FLAGS.batch_size, dataset=FLAGS.dataset)

    if FLAGS.is_train:
      analogy.train(max_iter=FLAGS.max_iter, alpha=FLAGS.alpha,
                    learning_rate=FLAGS.learning_rate, checkpoint_dir=FLAGS.checkpoint_dir)
    else:
      analogy.load(FLAGS.checkpoint_dir)

    analogy.test()

if __name__ == '__main__':
  tf.app.run()
