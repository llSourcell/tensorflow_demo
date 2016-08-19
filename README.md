# Tensorflow Demo

Overview
============
This project helps train a classifier to recognize handwritten character images [MNIST digits](http://yann.lecun.com/exdb/mnist/). It uses logistic regression as it's model and trains on a small MNIST dataset before testing the trained model on it. You can view the constructed data flow graph using Tensorboard in your browser after training. This is the code for TensorFlow in 5 Min on [Youtube](https://youtu.be/2FmcHiLCwTU)

Dependencies
============

* tensorflow (https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#pip-installation)

Use [pip](https://pypi.python.org/pypi/pip) to install any missing dependencies

Basic Usage
===========

1. Run ```python board.py``` to train the model. It will download & split the MNISt dataset into training and testing data. Then it will train a logistic regression model on the data. Lastly, it will test the trained model on the test set.

2. You can visualize the model in tensorboard by running ```tensorboard --logdir=LOCATION_ON_YOUR_COMPUTER```

That's it!

Credits
===========
Credit for the vast majority of code here goes to Google! I've merely created a wrapper around all of the important functions to get people started.
