Deep Visual Analogy-Making
==========================

Tensorflow implementation of [Deep Visual Analogy-Making](http://www-personal.umich.edu/~reedscot/nips2015.pdf). The matlab code and data of the paper can be found [here](http://www-personal.umich.edu/~reedscot/files/nips2015-analogy.tar.gz).

![model](./assets/model.png)

This implementation contains a deep network trained end-to-end to perform visual analogy making.


Prerequisites
-------------

- Python 2.7 or Python 3.3+
- [Tensorflow](https://www.tensorflow.org/)


Usage
-----

First, you need to download the dataset with:

    $ ./download.sh

To train a model with `sprite` (pixel character) dataset:

    $ python main.py --dataset sprite --is_train True

To test a model with `sprite` dataset:

    $ python main.py --dataset sprite


(in progress)


Author
------

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
