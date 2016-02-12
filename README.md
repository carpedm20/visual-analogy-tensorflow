Deep Visual Analogy-Making
==========================

Tensorflow implementation of [Deep Visual Analogy-Making](http://www-personal.umich.edu/~reedscot/nips2015.pdf). The matlab code of the paper can be found [here](http://www-personal.umich.edu/~reedscot/files/nips2015-analogy.tar.gz).

![model](assets/model.png)

This implementation contains a deep network trained end-to-end to perform visual analogy making with

1. Fully connected encoder & decoder networks
2. Analogy transformations by vector addition and deep networks (vector multiplication is not implemented)
3. Regularizer for manifold traversal transformations

This implementation conatins:

1. Analogy transformations of `shape` dataset
    - with objective for vector-addition-based analogies (L_add)
    - with objective for multiple fully connected layers (L_deep)
2. Animation transfer of `sprite` dataset
    - with objective for a disentangled feature representation (L_dis)
    - with objective for multiple softmax classifiers (L_{dis+cls})
3. Animation extrapoliation of `sprite` dataset
    - with objective for a disentangled feature representation (L_dis)
    - with objective for multiple softmax classifiers (L_{dis+cls})


Prerequisites
-------------

- Python 2.7 or Python 3.3+
- [Tensorflow](https://www.tensorflow.org/)
- [SciPy](http://www.scipy.org/)


Usage
-----

First, you need to download the dataset with:

    $ ./download.sh

To train a model with `sprite` (2d game character) dataset:

    $ python main.py --dataset sprite --is_train True

To test a model with `sprite` dataset:

    $ python main.py --dataset sprite


(in progress)


Results
-------

Result of analogy transformations of `shape` dataset with fully connected layers (L_deep).

- Change on angle

![training in progress](./assets/rotate_160212.png)

- Change on scale

![training in progress](./assets/scale_160212.png)

- Change on x position

![training in progress](./assets/xpos_160212.png)

- Change on y position

![training in progress](./assets/ypos_160212.png)

*Reference*, *output*, *query*, *target*, *prediction*, and *difference*, in order from top to bottom in each image.


Result of analogy transformations of `sprite` dataset.

(in progress)


Training details
----------------

(in progress)

![training in progress](./assets/shape_loss_160211.png)



Reference
---------

- [NIPS 2015 slide](http://www-personal.umich.edu/~reedscot/files/nips2015-analogy-slides.pptx)


Author
------

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
