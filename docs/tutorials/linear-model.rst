############
Linear Model
############

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
observed neural activity - in this example, spike counts. These arrays have the
same number of rows, corresponding to the number of time points in the dataset.
We'll assume that each row of ''xstim'' contains the brightness values of a 
full-screen stimulus 10 time points into the past from that time point. Each row
of ''robs'' contains the spike count of all recorded units at that time.

We can estimate the linear model, or STA, of each recorded unit by taking the dot 
product of ''xstim'' and ''robs'' which is here equivalent to averaging across the 
frames preceding each spike.
We'll take the dot product with

>>> STA = np.matmul(np.transpose(robs), xstim)


TODO

