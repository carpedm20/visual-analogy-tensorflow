Deep Visual Analogy-Making
==========================

Tensorflow implementation of [Deep Visual Analogy-Making](http://www-personal.umich.edu/~reedscot/nips2015.pdf). The matlab code of the paper can be found [here](http://www-personal.umich.edu/~reedscot/files/nips2015-analogy.tar.gz).

![model](./assets/model.png)

This implementation contains a deep network trained end-to-end to perform visual analogy making with

1. Fully connected encoder & decoder networks
2. Analogy transformations by vector addition and deep networks (vector multiplication is not implemented)
3. Regularizer to manifold traversal transformations


Prerequisites
-------------

- Python 2.7 or Python 3.3+
- [Tensorflow](https://www.tensorflow.org/)


Usage
-----

First, you need to download the dataset with:

    $ ./download.sh

To train a model with `sprite` (2d game character) dataset:

    $ python main.py --dataset sprite --is_train True

To test a model with `sprite` dataset:

    $ python main.py --dataset sprite


(in progress)


Author
------

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
