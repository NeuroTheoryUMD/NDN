#####################
Linear Model Tutorial
#####################

Let's start with the basics. First we're going to import all of the necessary
modules. We'll do this once now, and ignore for future tutorials.

Import numpy and tensorflow::

>>> import numpy as np
>>> import tensorflow as tf

Next we're going to tell the python kernel where the NDN code is located, and 
import the necessary modules::

>>> import sys
>>> sys.path.insert(0, '/path/to/code/')
>>> import NDN.NDN.NDN as NDN
>>> import NDN.NDNutils as NDNutils

Now we're going to assume that we have two numpy arrays, one called ``xstim`` 
containing the stimulus values, and another called ``robs`` containing the 
observed neural activity.

TODO

