# Tensorflow Demo

Overview
============
This project helps train a classifier to recognize handwritten character images [MNIST digits](http://yann.lecun.com/exdb/mnist/). It uses logistic regression as it's model and trains on a small MNIST dataset before testing the trained model on it. You can view the constructed data flow graph using Tensorboard in your browser after training. This is the code for TensorFlow in 5 Min on [Youtube](https://youtu.be/2FmcHiLCwTU)
<p>
<img src="https://camo.githubusercontent.com/d440ac2eee1cb3ea33340a2c5f6f15a0878e9275/687474703a2f2f692e7974696d672e636f6d2f76692f3051493378675875422d512f687164656661756c742e6a7067" height="220px" align="left">
</p>

Dependencies
============

TensorFlow (https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#pip-installation)

Use [pip](https://pypi.python.org/pypi/pip) to install any missing dependencies

Basic Usage
===========

0. Change Line 56 in board.py to a location on your computer

1. Run ```python board.py``` to train the model. It will download & split the MNISt dataset into training and testing data. Then it will train a logistic regression model on the data. Lastly, it will test the trained model on the test set.

2. You can visualize the model in tensorboard by running ```tensorboard --logdir=LOCATION_ON_YOUR_COMPUTER```

That's it!

Credits
===========
Credit for the vast majority of code here goes to Google! I've merely created a wrapper around all of the important functions to get people started.
