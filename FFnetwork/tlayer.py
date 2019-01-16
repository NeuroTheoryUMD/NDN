"""Basic temporal-layer definitions"""

from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
from .regularization import Regularization
from .regularization import SepRegularization

from .layer import Layer


#
# class TLayer(Layer):
#     """Implementation of calcium tent layer
#
#     Attributes:
#         filter_width (int): time spread
#         batch_size (int): the batch size is explicitly needed for this computation
#
#     """
#
#     def __init__(
#             self,
#             scope=None,
#             nlags=None,
#             input_dims=None,  # this can be a list up to 3-dimensions
#             output_dims=None,
#             num_filters=None,
#             batch_size=None,
#             time_spread=None,
#             activation_func='relu',
#             normalize_weights=0,
#             weights_initializer='trunc_normal',
#             biases_initializer='zeros',
#             reg_initializer=None,
#             num_inh=0,
#             pos_constraint=True,
#             log_activations=False):
#         """Constructor for convLayer class
#
#         Args:
#             scope (str): name scope for variables and operations in layer
#             input_dims (int or list of ints): dimensions of input data
#             num_filters (int): number of convolutional filters in layer
#             filter_dims (int or list of ints): dimensions of input data
#             shift_spacing (int): stride of convolution operation
#             activation_func (str, optional): pointwise function applied to
#                 output of affine transformation
#                 ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' |
#                 'elu' | 'quad'
#             normalize_weights (int): 1 to normalize weights 0 otherwise
#                 [0] | 1
#             weights_initializer (str, optional): initializer for the weights
#                 ['trunc_normal'] | 'normal' | 'zeros'
#             biases_initializer (str, optional): initializer for the biases
#                 'trunc_normal' | 'normal' | ['zeros']
#             reg_initializer (dict, optional): see Regularizer docs for info
#             num_inh (int, optional): number of inhibitory units in layer
#             pos_constraint (bool, optional): True to constrain layer weights to
#                 be positive
#             log_activations (bool, optional): True to use tf.summary on layer
#                 activations
#
#         Raises:
#             ValueError: If `pos_constraint` is `True`
#
#         """
#         self.batch_size = batch_size
#         self.time_spread = time_spread
#
#         # TODO: maybe you will need to fix num_filters later
#         num_filters = 1
#
#         # Process stim and filter dimensions
#         # (potentially both passed in as num_inputs list)
#         if isinstance(input_dims, list):
#             while len(input_dims) < 3:
#                 input_dims.append(1)
#         else:
#             # assume 1-dimensional (space)
#             input_dims = [1, input_dims, 1]
#
#         super(TLayer, self).__init__(
#             scope=scope,
#             nlags=nlags,
#             input_dims=input_dims,
#             output_dims=output_dims,  # Note difference from layer
#             filter_dims=None, #TODO: fix this for reg
#             my_num_inputs=batch_size,
#             my_num_outputs=batch_size,  # effectively
#             activation_func=activation_func,
#             normalize_weights=normalize_weights,
#             weights_initializer=weights_initializer,
#             biases_initializer=biases_initializer,
#             num_inh=num_inh,
#             pos_constraint=pos_constraint,
#             log_activations=log_activations)
#
#         # TODO: remember, you can use this part to overwrite anything
#        # self.output_dims = input_dims
#       #  self.num_filters = input_dims[0]
#
#     # END CaTentLayer.__init__
#
#     def build_graph(self, inputs, params_dict=None):
#
#         with tf.name_scope(self.scope):
#             self._define_layer_variables()
#
#             # only upper triangular part is needed
#             weights_tri = tf.matrix_band_part(self.weights_var, 0, -1)
#
#             # if normalization... (maybe change layer, this is highly inefficient)
#             ws = tf.transpose(self._normalize_weights(tf.transpose(weights_tri)))
#
#             if self.pos_constraint:
#                 pre = tf.add(tf.matmul(tf.maximum(0.0, ws), inputs), tf.transpose(self.biases_var))
#             else:
#                 pre = tf.add(tf.matmul(ws, inputs), tf.transpose(self.biases_var))
#
#             if self.ei_mask_var is not None:
#                 post = tf.multiply(self.activation_func(pre), self.ei_mask_var)
#             else:
#                 post = self.activation_func(pre)
#
#             self.outputs = post
#
#         if self.log:
#             tf.summary.histogram('act_pre', pre)
#             tf.summary.histogram('act_post', post)
#     # END TLayer.build_graph

class CaTentLayer(Layer):
    """Implementation of calcium tent layer

    Attributes:
        filter_width (int): time spread
        batch_size (int): the batch size is explicitly needed for this computation

    """

    def __init__(
            self,
            scope=None,
            nlags=None,
            input_dims=None,  # this can be a list up to 3-dimensions
            output_dims=None,
            num_filters=None,
            filter_width=None,  # this can be a list up to 3-dimensions
            batch_size=None,
            activation_func='lin',
            normalize_weights=True,
            weights_initializer='normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=True,
            log_activations=False):
        """Constructor for convLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int or list of ints): dimensions of input data
            num_filters (int): number of convolutional filters in layer
            filter_dims (int or list of ints): dimensions of input data
            shift_spacing (int): stride of convolution operation
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' |
                'elu' | 'quad'
            normalize_weights (int): 1 to normalize weights 0 otherwise
                [0] | 1
            weights_initializer (str, optional): initializer for the weights
                ['trunc_normal'] | 'normal' | 'zeros'
            biases_initializer (str, optional): initializer for the biases
                'trunc_normal' | 'normal' | ['zeros']
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (bool, optional): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations

        Raises:
            ValueError: If `pos_constraint` is `True`

        """

        self.batch_size = batch_size
        self.filter_width = filter_width

        # Process stim and filter dimensions
        # (potentially both passed in as num_inputs list)
        if isinstance(input_dims, list):
            while len(input_dims) < 3:
                input_dims.append(1)
        else:
            # assume 1-dimensional (space)
            input_dims = [1, input_dims, 1]

        # If output dimensions already established, just strip out num_filters
      #  if isinstance(num_filters, list):
      #      num_filters = num_filters[0]

        # TODO: how to specify num filters...
        if num_filters > 1:
            num_filters = input_dims[1]

        super(CaTentLayer, self).__init__(
            scope=scope,
            nlags=nlags,
            input_dims=input_dims,
            output_dims=output_dims,  # Note difference from layer
            my_num_inputs=filter_width,
            my_num_outputs=num_filters,
            activation_func=activation_func,
            normalize_weights=normalize_weights,
            weights_initializer=weights_initializer,
            biases_initializer=biases_initializer,
            reg_initializer=reg_initializer,
            num_inh=num_inh,
            pos_constraint=pos_constraint,
            log_activations=log_activations)

        self.output_dims = input_dims

        # ei_mask not useful at the moment
        self.ei_mask_var = None

        # make nc biases instead of only 1
        bias_dims = (1, self.output_dims[1])
        init_biases = np.zeros(shape=bias_dims, dtype='float32')
        self.biases = init_biases

        self.reg = Regularization(
            input_dims=[filter_width, 1, 1],
            num_outputs=num_filters,
            vals=reg_initializer)

    # END CaTentLayer.__init__

    def build_graph(self, inputs, params_dict=None, use_dropout=False):

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            # make shaped input
            shaped_input = tf.reshape(tf.transpose(inputs), [-1, self.batch_size, 1, 1])

            if self.pos_constraint is not None:
                w_p = tf.maximum(self.weights_var, 0.0)
            else:
                w_p = self.weights_var

            if self.normalize_weights > 0:
                w_pn = tf.nn.l2_normalize(w_p, axis=0)
            else:
                w_pn = w_p

            padding = tf.constant([[0, self.filter_width], [0, 0]])
            padded_filt = tf.pad(w_pn, padding)
            shaped_padded_filt = tf.reshape(padded_filt, [2*self.filter_width, 1, 1, self.num_filters])

            # convolve
            strides = [1, 1, 1, 1]
            _pre = tf.nn.conv2d(shaped_input, shaped_padded_filt, strides, padding='SAME')

            # transpose, squeeze to final shape
            # both cases will produce _pre_final_shape.shape ---> (batch_size, nc)
            if self.num_filters > 1:
                _pre_final_shape = tf.linalg.diag_part(tf.transpose(tf.squeeze(_pre, axis=2), [1, 0, 2]))
            else:
                # single filter
                _pre_final_shape = tf.transpose(tf.squeeze(_pre, axis=[2, 3]))

            pre = tf.add(_pre_final_shape, self.biases_var)
            self.outputs = self._apply_act_func(pre)

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END CaTentLayer.build_graph


# TODO: fix input/output dims so that multiple consecutive TLayers are possible
class TLayer(Layer):
    """Implementation of a temporal layer

    Attributes:
        filter_width (int): time spread
        batch_size (int): the batch size is explicitly needed for this computation

    """

    def __init__(
            self,
            scope=None,
            nlags=None,
            input_dims=None,  # this can be a list up to 3-dimensions
            output_dims=None,
            num_filters=None,
            filter_width=None,  # this can be a list up to 3-dimensions
            batch_size=None,
            dilation=1,
            activation_func='lin',
            normalize_weights=True,
            weights_initializer='trunc_normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=False,
            log_activations=False):
        """Constructor for convLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int or list of ints): dimensions of input data
            num_filters (int): number of convolutional filters in layer
            filter_dims (int or list of ints): dimensions of input data
            shift_spacing (int): stride of convolution operation
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' |
                'elu' | 'quad'
            normalize_weights (int): 1 to normalize weights 0 otherwise
                [0] | 1
            weights_initializer (str, optional): initializer for the weights
                ['trunc_normal'] | 'normal' | 'zeros'
            biases_initializer (str, optional): initializer for the biases
                'trunc_normal' | 'normal' | ['zeros']
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (bool, optional): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations

        Raises:
            ValueError: If `pos_constraint` is `True`

        """

        self.batch_size = batch_size
        self.filter_width = filter_width

        # Process stim and filter dimensions
        # (potentially both passed in as num_inputs list)
     #  if isinstance(input_dims, list):
     #       while len(input_dims) < 3:
     #           input_dims.append(1)
     #   else:
     #       # assume 1-dimensional (space)
     #       input_dims = [1, input_dims, 1]

        # If output dimensions already established, just strip out num_filters
      #  if isinstance(num_filters, list):
      #      num_filters = num_filters[0]

        super(TLayer, self).__init__(
            scope=scope,
            nlags=nlags,
            input_dims=input_dims,
            output_dims=output_dims,  # Note difference from layer
            my_num_inputs=filter_width,
            my_num_outputs=num_filters,
            activation_func=activation_func,
            normalize_weights=normalize_weights,
            weights_initializer=weights_initializer,
            biases_initializer=biases_initializer,
            reg_initializer=reg_initializer,
            num_inh=num_inh,
            pos_constraint=pos_constraint,
            log_activations=log_activations)

   #     self.output_dims = deepcopy(input_dims)
        self.output_dims[0] = self.num_filters

        self.dilation = dilation

        # ei_mask not useful at the moment
        self.ei_mask_var = None

        self.reg = Regularization(
            input_dims=[filter_width, 1, 1],
            num_outputs=num_filters,
            vals=reg_initializer)

    # END TLayer.__init__

    def build_graph(self, inputs, params_dict=None, use_dropout=False):

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            # make shaped input
            shaped_input = tf.reshape(tf.transpose(inputs),
                                      [self.input_dims[1]*self.input_dims[2], self.batch_size, 1, 1])

            if self.pos_constraint is not None:
                w_p = tf.maximum(self.weights_var, 0.0)
            else:
                w_p = self.weights_var

            if self.normalize_weights > 0:
                w_pn = tf.nn.l2_normalize(w_p, axis=0)
            else:
                w_pn = w_p

            padding = tf.constant([[0, self.filter_width], [0, 0]])
            padded_filt = tf.pad(w_pn, padding)
            shaped_padded_filt = tf.reshape(padded_filt, [2*self.filter_width, 1, 1, self.num_filters])

            # convolve
            strides = [1, 1, 1, 1]
            dilations = [1, self.dilation, 1, 1]
            _pre = tf.nn.conv2d(shaped_input, shaped_padded_filt, strides, dilations=dilations, padding='SAME')
            pre = tf.add(_pre, self.biases_var)
            post = self._apply_act_func(pre)

            self.outputs = tf.reshape(tf.transpose(post, [1, 0, 2, 3]), (self.batch_size, -1))

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END CaTentLayer.build_graph


class NoRollCaTentLayer(Layer):
    """Implementation of calcium tent layer

    Attributes:
        filter_width (int): time spread
        batch_size (int): the batch size is explicitly needed for this computation

    """

    def __init__(
            self,
            scope=None,
            nlags=None,
            input_dims=None,  # this can be a list up to 3-dimensions
            output_dims=None,
            num_filters=None,
            filter_width=None,  # this can be a list up to 3-dimensions
            batch_size=None,
            activation_func='lin',
            normalize_weights=0,
            weights_initializer='trunc_normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=True,
            log_activations=False):
        """Constructor for convLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int or list of ints): dimensions of input data
            num_filters (int): number of convolutional filters in layer
            filter_dims (int or list of ints): dimensions of input data
            shift_spacing (int): stride of convolution operation
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' |
                'elu' | 'quad'
            normalize_weights (int): 1 to normalize weights 0 otherwise
                [0] | 1
            weights_initializer (str, optional): initializer for the weights
                ['trunc_normal'] | 'normal' | 'zeros'
            biases_initializer (str, optional): initializer for the biases
                'trunc_normal' | 'normal' | ['zeros']
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (bool, optional): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations

        Raises:
            ValueError: If `pos_constraint` is `True`

        """

        if filter_width is None:
            filter_width = 2*batch_size

        self.batch_size = batch_size
        self.filter_width = filter_width

        # Process stim and filter dimensions
        # (potentially both passed in as num_inputs list)
        if isinstance(input_dims, list):
            while len(input_dims) < 3:
                input_dims.append(1)
        else:
            # assume 1-dimensional (space)
            input_dims = [1, input_dims, 1]

        # If output dimensions already established, just strip out num_filters
      #  if isinstance(num_filters, list):
      #      num_filters = num_filters[0]

        # TODO: how to specify num filters...
        if num_filters > 1:
            num_filters = input_dims[1]

        super(NoRollCaTentLayer, self).__init__(
            scope=scope,
            nlags=nlags,
            input_dims=input_dims,
            output_dims=output_dims,  # Note difference from layer
            my_num_inputs=filter_width,
            my_num_outputs=num_filters,
            activation_func=activation_func,
            normalize_weights=normalize_weights,
            weights_initializer=weights_initializer,
            biases_initializer=biases_initializer,
            reg_initializer=reg_initializer,
            num_inh=num_inh,
            pos_constraint=pos_constraint,
            log_activations=log_activations)

        self.output_dims = input_dims

    # END CaTentLayer.__init__

    def build_graph(self, inputs, params_dict=None):

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            # make shaped input
            shaped_input = tf.reshape(tf.transpose(inputs), [self.input_dims[1], self.batch_size, 1, 1])

            # make shaped filt
            conv_filt_shape = [self.filter_width, 1, 1, self.num_filters]

            if self.normalize_weights > 0:
                wnorms = tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(self.weights_var), axis=0)), 1e-8)
                shaped_filt = tf.reshape(tf.divide(self.weights_var, wnorms), conv_filt_shape)
            else:
                shaped_filt = tf.reshape(self.weights_var, conv_filt_shape)

            # convolve
            strides = [1, 1, 1, 1]
            if self.pos_constraint:
                pre = tf.nn.conv2d(shaped_input, tf.maximum(0.0, shaped_filt), strides, padding='SAME')
            else:
                pre = tf.nn.conv2d(shaped_input, shaped_filt, strides, padding='SAME')

            # from pre to post
            if self.ei_mask_var is not None:
                post = tf.multiply(
                    self.activation_func(tf.add(pre, self.biases_var)),
                    self.ei_mask_var)
            else:
                post = self.activation_func(tf.add(pre, self.biases_var))

            # this produces shape (batch_size, nc, num_filts)
            # after matrix_diag_part we have diagonal part ---> shape will be (batch_size, nc)
            self.outputs = tf.matrix_diag_part(tf.transpose(tf.squeeze(post, axis=2), [1, 0, 2]))

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END NoRollCaTentLayer.build_graph
