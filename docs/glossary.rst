########
Glossary
########

Network Types
#############

* **FF Network**

* **Side Network**

Layer Types
###########

* **Normal Layer** - fully-connected layer

* **Convolution Layer** - convolutional layer

* **Separable Layer** - decomposes a fully-connected layer after a convolutional layer into a spatial component and a filter component, dramatically saving the number of weights. See the `original paper`_.

* **Additive Layer** - combines input from multiple input streams.

Regularization Types
####################

The NDN package applies regularization to model parameters at the level of 
individual layers, and only affects the weights (not the bias terms). To describe
the different regularization options available, we define the vector 
:math:`\mathbf{w}` to be the vectorized weight matrix, and :math:`\lambda` to be the
corresponding regularization hyperparameter that controls the contribution of 
the regularization term to the cost function. The following is a brief description
of each option, and the corresponding penalty :math:`L` that is added to 
the model's cost function.

**Standard Regularizers**

* **l2** - standard l2 regularization on model parameters that penalizes weights from becoming too large.
.. math::
    L = \lambda \sum_{i=1}^N w_i^2 

* **l1** - standard l1 regularization on model parameters to encourage sparseness. 
.. math::
    L = \lambda \sum_{i=1}^N | w_i |

**Smoothing Regularizers**

The following regularization options are all variants of `Tikhonov regularization`_,
which adds

.. math::
    L = \lambda \| T * \mathbf{w} \|_2^2

as a penalty term to the model's cost function. 
:math:`T` is an appropriately-defined Laplacian operator, which penalizes the second
derivative of model parameters in either spatial or temporal domain, or both. The
effect of this term is to smooth model parameters in desired domain.

* **d2t** - encourages smoothness in the temporal domain.

* **d2x** - encourages smoothness in the spatial domain.

* **d2xt** - encourages smoothness in both the temporal and spatial domains.

**Sparsifying Regularizers**

The following regularization options are related to `Tikhonov regularization`_,
and add

.. math::
    L = \lambda \| T * \mathbf{w}^2 \|_2^2

as a penalty term to the model's cost function, where :math:`\mathbf{w}^2` represents
the element-wise squaring of :math:`\mathbf{w}^2`. 
:math:`T` is a matrix of 1's, with 0's on the diagonal.

* **max_filt** - encourages the output from a single filter (but an arbitrary number of spatial locations) to project to the next layer. 

* **max_space** - encourages the output from a single spatial position (but an arbitrary number of filters) to project to the next layer. 

* **max** - encourages the output from a single filter and single spatial location to project to the next layer.

**Other Regularizers**

* **center** - encourages centered convolutional filters

* **norm2** - encourages the norm of a weight matrix to be near 1.

.. _`Tikhonov regularization`: https://en.wikipedia.org/wiki/Tikhonov_regularization
.. _`original paper`: http://papers.nips.cc/paper/6942-neural-system-identification-for-large-populations-separating-what-and-where
