"""Basic layer definitions"""

from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
from .regularization import Regularization
from .regularization import SepRegularization


class Layer(object):
    """Implementation of fully connected neural network layer

    Attributes:
        scope (str): name scope for variables and operations in layer
        input_dims (list): inputs to layer
        output_dims (list): outputs of layer
        outputs (tf Tensor): output of layer
        weights_ph (tf placeholder): placeholder for weights in layer
        biases_ph (tf placeholder): placeholder for biases in layer
        weights_var (tf Tensor): weights in layer
        biases_var (tf Tensor): biases in layer
        weights (numpy array): shadow variable of `weights_var` that allows for 
            easier manipulation outside of tf sessions
        biases (numpy array): shadow variable of `biases_var` that allows for 
            easier manipulation outside of tf sessions
        activation_func (tf activation function): activation function in layer
        reg (Regularization object): holds regularizations values and matrices
            (as tf constants) for layer
        ei_mask_var (tf constant): mask of +/-1s to multiply output of layer
        ei_mask (list): mask of +/-1s to multiply output of layer; shadows 
            `ei_mask_tf` for easier manipulation outside of tf sessions
        pos_constraint (bool): positivity constraint on weights in layer
        num_filters (int): equal to output_dims
        filter_dims (list of ints): equal to input_dims
        normalize_weights (int): defines normalization type for weights in 
            layer
        log (bool): use tf summary writers on layer output

    """

    def __init__(
            self,
            scope=None,
            input_dims=None,  # this can be a list up to 3-dimensions
            filter_dims=None,
            output_dims=None,
            activation_func='relu',
            normalize_weights=0,
            weights_initializer='trunc_normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=False,
            log_activations=False):
        """Constructor for Layer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int or list of ints): dimensions of input data
            filter_dims (int or list of ints): dimensions of input data
            output_dims (int or list of ints): dimensions of output data
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
            TypeError: If `variable_scope` is not specified
            TypeError: If `input_dims` is not specified
            TypeError: If `output_dims` is not specified
            ValueError: If `activation_func` is not a valid string
            ValueError: If `num_inh` is greater than total number of units
            ValueError: If `weights_initializer` is not a valid string
            ValueError: If `biases_initializer` is not a valid string

        """

        # check for required inputs
        if scope is None:
            raise TypeError('Must specify layer scope')
        if input_dims is None or output_dims is None:
            raise TypeError('Must specify both input and output dimensions')

        self.scope = scope

        # Make input, output, and filter sizes explicit
        if isinstance(input_dims, list):
            while len(input_dims) < 3:
                input_dims.append(1)
        else:
            input_dims = [1, input_dims, 1]
        if isinstance(output_dims,list):
            while len(output_dims) < 3:
                output_dims.append(1)
            num_outputs = output_dims[0]*output_dims[1]*output_dims[2]
        else:
            num_outputs = output_dims
            output_dims = [1, output_dims, 1]

        self.input_dims = input_dims
        self.output_dims = output_dims
        # default to have N filts for N outputs in base layer class
        self.num_filters = num_outputs
        if filter_dims is None:
            filter_dims = input_dims
        num_inputs = filter_dims[0] * filter_dims[1] * filter_dims[2]
        self.filter_dims = filter_dims

        # resolve activation function string
        if activation_func == 'relu':
            self.activation_func = tf.nn.relu
        elif activation_func == 'sigmoid':
            self.activation_func = tf.sigmoid
        elif activation_func == 'tanh':
            self.activation_func = tf.tanh
        elif activation_func == 'lin':
            self.activation_func = tf.identity
        elif activation_func == 'linear':
            self.activation_func = tf.identity
        elif activation_func == 'softplus':
            self.activation_func = tf.nn.softplus
        elif activation_func == 'quad':
            self.activation_func = tf.square
        elif activation_func == 'elu':
            self.activation_func = tf.nn.elu
        elif activation_func == 'exp':
            self.activation_func = tf.exp
        else:
            raise ValueError('Invalid activation function ''%s''' %
                             activation_func)

        # create excitatory/inhibitory mask
        if num_inh > num_outputs:
            raise ValueError('Too many inhibitory units designated')
        self.ei_mask = [1] * (num_outputs - num_inh) + [-1] * num_inh

        # save positivity constraint on weights
        self.pos_constraint = pos_constraint
        self.normalize_weights = normalize_weights

        # use tf's summary writer to save layer activation histograms
        if log_activations:
            self.log = True
        else:
            self.log = False

        # Set up layer regularization
        self.reg = Regularization(
            input_dims=filter_dims,
            num_outputs=num_outputs,
            vals=reg_initializer)

        # Initialize weight values
        weight_dims = (num_inputs, num_outputs)
        if weights_initializer == 'trunc_normal':
            init_weights = np.random.normal(size=weight_dims, scale=0.1)
        elif weights_initializer == 'normal':
            init_weights = np.random.normal(size=weight_dims, scale=0.1)
        elif weights_initializer == 'zeros':
            init_weights = np.zeros(shape=weight_dims, dtype='float32')
        else:
            raise ValueError('Invalid weights_initializer ''%s''' %
                             weights_initializer)
        if pos_constraint:
            init_weights = np.maximum(init_weights, 0)
        if normalize_weights > 0:
            init_weights = np.divide(
                init_weights,
                np.sqrt(np.sum(np.square(init_weights), axis=0)))

        # Initialize numpy array that will feed placeholder
        self.weights = init_weights.astype('float32')

        # Initialize bias values
        bias_dims = (1, num_outputs)
        if biases_initializer == 'trunc_normal':
            init_biases = np.random.normal(size=bias_dims, scale=0.1)
        elif biases_initializer == 'normal':
            init_biases = np.random.normal(size=bias_dims, scale=0.1)
        elif biases_initializer == 'zeros':
            init_biases = np.zeros(shape=bias_dims, dtype='float32')
        else:
            raise ValueError('Invalid biases_initializer ''%s''' %
                             biases_initializer)
        # Initialize numpy array that will feed placeholder
        self.biases = init_biases.astype('float32')

        # Define tensorflow variables as placeholders
        self.weights_ph = None
        self.weights_var = None
        self.biases_ph = None
        self.biases_var = None
        self.outputs = None

    # END Layer.__init__

    def _define_layer_variables(self):
        # Define tensor-flow versions of variables (placeholder and variables)

        with tf.name_scope('weights_init'):
            self.weights_ph = tf.placeholder_with_default(
                self.weights,
                shape=self.weights.shape,
                name='weights_ph')
            self.weights_var = tf.Variable(
                self.weights_ph,
                dtype=tf.float32,
                name='weights_var')

        # Initialize biases placeholder/variable
        with tf.name_scope('biases_init'):
            self.biases_ph = tf.placeholder_with_default(
                self.biases,
                shape=self.biases.shape,
                name='biases_ph')
            self.biases_var = tf.Variable(
                self.biases_ph,
                dtype=tf.float32,
                name='biases_var')

        # Check for need of ei_mask
        if np.sum(self.ei_mask) < len(self.ei_mask):
            self.ei_mask_var = tf.constant(
                self.ei_mask, dtype=tf.float32, name='ei_mask')
        else:
            self.ei_mask_var = None

    # END Layer._define_layer_variables

    def build_graph(self, inputs, params_dict=None):

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            # Define computation
            if self.normalize_weights > 0:
                wnorms = tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(self.weights_var), axis=0)), 1e-8)
                ws = tf.divide(self.weights_var, wnorms)
            else:
                #ws = tf.identity(self.weights_var)
                ws = self.weights_var

            if self.pos_constraint:
                pre = tf.add(tf.matmul(
                    inputs,
                    tf.maximum(0.0, ws)), self.biases_var)
            else:
                pre = tf.add(
                    tf.matmul(inputs, ws), self.biases_var)

            if self.ei_mask_var is not None:
                post = tf.multiply(self.activation_func(pre), self.ei_mask_var)
            else:
                post = self.activation_func(pre)

            self.outputs = post

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END layer.build_graph

    def assign_layer_params(self, sess):
        """Read weights/biases in numpy arrays into tf Variables"""
        sess.run(
            [self.weights_var.initializer, self.biases_var.initializer],
            feed_dict={self.weights_ph: self.weights,
                       self.biases_ph: self.biases})

    def write_layer_params(self, sess):
        """Write weights/biases in tf Variables to numpy arrays"""

        self.weights = sess.run(self.weights_var)

        if self.pos_constraint:
            self.weights = np.maximum(self.weights, 0)

        if self.normalize_weights > 0:
            wnorm = np.sqrt(np.sum(np.square(self.weights), axis=0))
            wnorm[np.where(wnorm == 0)] = 1
            self.weights = np.divide(self.weights, wnorm)

        self.biases = sess.run(self.biases_var)

    def define_regularization_loss(self):
        """Wrapper function for building regularization portion of graph"""
        with tf.name_scope(self.scope):
            return self.reg.define_reg_loss(self.weights_var)

    def set_regularization(self, reg_type, reg_val):
        """Wrapper function for setting regularization"""
        return self.reg.set_reg_val(reg_type, reg_val)

    def assign_reg_vals(self, sess):
        """Wrapper function for assigning regularization values"""
        self.reg.assign_reg_vals(sess)

    def get_reg_pen(self, sess):
        """Wrapper function for returning regularization penalty dict"""
        return self.reg.get_reg_penalty(sess)


class ConvLayer(Layer):
    """Implementation of convolutional layer

    Attributes:
        shift_spacing (int): stride of convolution operation
        num_shifts (int): number of shifts in horizontal and vertical 
            directions for convolution operation
            
    """

    def __init__(
            self,
            scope=None,
            input_dims=None,   # this can be a list up to 3-dimensions
            num_filters=None,
            filter_dims=None,  # this can be a list up to 3-dimensions
            shift_spacing=1,
            activation_func='relu',
            normalize_weights=0,
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

        # Error checking
        if pos_constraint is True:
            raise ValueError(
                'No positive constraint should be applied to this layer')

        # Process stim and filter dimensions
        # (potentially both passed in as num_inputs list)
        if isinstance(input_dims, list):
            while len(input_dims) < 3:
                input_dims.append(1)
        else:
            # assume 1-dimensional (space)
            input_dims = [1, input_dims, 1]

        if filter_dims is None:
            filter_dims = input_dims
        else:
            if isinstance(filter_dims, list):
                while len(filter_dims) < 3:
                    filter_dims.extend(1)
            else:
                filter_dims = [filter_dims, 1, 1]

        # If output dimensions already established, just strip out num_filters
        if isinstance(num_filters, list):
            num_filters = num_filters[0]

        # Calculate number of shifts (for output)
        num_shifts = [1, 1]
        if input_dims[1] > 1:
            num_shifts[0] = int(np.floor(input_dims[1]/shift_spacing))
        if input_dims[2] > 1:
            num_shifts[1] = int(np.floor(input_dims[2]/shift_spacing))

        super(ConvLayer, self).__init__(
                scope=scope,
                input_dims=input_dims,
                filter_dims=filter_dims,
                output_dims=num_filters,   # Note difference from layer
                activation_func=activation_func,
                normalize_weights=normalize_weights,
                weights_initializer=weights_initializer,
                biases_initializer=biases_initializer,
                reg_initializer=reg_initializer,
                num_inh=num_inh,
                pos_constraint=False,        # Note difference from layer
                log_activations=log_activations)

        # ConvLayer-specific properties
        self.shift_spacing = shift_spacing
        self.num_shifts = num_shifts
        # Changes in properties from Layer - note this is implicitly
        # multi-dimensional
        self.output_dims = [num_filters] + num_shifts

    # END ConvLayer.__init__

    def build_graph(self, inputs, params_dict=None):

        assert params_dict is not None, 'Incorrect siLayer initialization.'
        # Unfold siLayer-specific parameters for building graph
        filter_size = self.filter_dims
        num_shifts = self.num_shifts

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            # Computation performed in the layer
            # Reshape of inputs (4-D):
            input_dims = [-1, self.input_dims[2], self.input_dims[1],
                          self.input_dims[0]]
            # this is reverse-order from Matlab:
            # [space-2, space-1, lags, and num_examples]
            shaped_input = tf.reshape(inputs, input_dims)

            # Reshape weights (4:D:
            conv_filter_dims = [filter_size[2], filter_size[1], filter_size[0],
                                self.num_filters]

            if self.normalize_weights > 0:
                wnorms = tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(self.weights_var), axis=0)), 1e-8)
                ws_conv = tf.reshape(tf.divide(self.weights_var, wnorms), conv_filter_dims)
            else:
                ws_conv = tf.reshape(self.weights_var, conv_filter_dims)
                # this is reverse-order from Matlab:
                # [space-2, space-1, lags] and num_filters is explicitly last dim

            # Make strides list
            strides = [1, 1, 1, 1]
            if conv_filter_dims[1] > 1:
                strides[1]=self.shift_spacing
            if conv_filter_dims[2] > 1:
                strides[2] = self.shift_spacing

            pre = tf.nn.conv2d(shaped_input, ws_conv, strides, padding='SAME')

            if self.ei_mask_var is not None:
                post = tf.multiply(
                    tf.add(self.activation_func(pre), self.biases_var),
                    self.ei_mask_var)
            else:
                post = tf.add(self.activation_func(pre), self.biases_var)

            self.outputs = tf.reshape(
                post, [-1, self.num_filters * num_shifts[0] * num_shifts[1]])

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END ConvLayer.build_graph


class SepLayer(Layer):
    """Implementation of separable neural network layer; see 
    http://papers.nips.cc/paper/6942-neural-system-identification-for-large-populations-separating-what-and-where
    for more info

    """

    def __init__(
            self,
            scope=None,
            input_dims=None, # this can be a list up to 3-dimensions
            output_dims=None,
            activation_func='relu',
            normalize_weights=0,
            weights_initializer='trunc_normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=False,
            log_activations=False):
        """Constructor for sepLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int): dimensions of input data
            output_dims (int): dimensions of output data
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' | 
                'elu' | 'quad'
            normalize_weights (int): type of normalization to apply to the 
                weights. Default [0] is to normalize across the first dimension 
                (time/filters), but '1' will normalize across spatial 
                dimensions instead, and '2' will normalize both
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

        """

        # Process stim and filter dimensions
        # (potentially both passed in as num_inputs list)
        if isinstance(input_dims, list):
            while len(input_dims) < 3:
                input_dims.append(1)
        else:
            input_dims = [1, input_dims, 1]  # assume 1-dimensional (space)

        # Determine filter dimensions (first dim + space_dims
        num_space = input_dims[1]*input_dims[2]
        filter_dims = [input_dims[0]+num_space, 1, 1]

        super(SepLayer, self).__init__(
                scope=scope,
                input_dims=input_dims,
                filter_dims=filter_dims,
                output_dims=output_dims,
                activation_func=activation_func,
                normalize_weights=normalize_weights,
                weights_initializer=weights_initializer,
                biases_initializer=biases_initializer,
                reg_initializer=reg_initializer,
                num_inh=num_inh,
                pos_constraint=pos_constraint,
                log_activations=log_activations)

        # Redefine specialized Regularization object to overwrite default
        self.reg = SepRegularization(
            input_dims=input_dims,
            num_outputs=self.reg.num_outputs,
            vals=reg_initializer)
    # END SepLayer.__init__

    def build_graph(self, inputs, params_dict=None):

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            # Section weights into first dimension and space
            kts = tf.slice(self.weights_var, [0, 0],
                           [self.input_dims[0], self.num_filters] )

            ksp = tf.slice(self.weights_var, [self.input_dims[0], 0],
                           [self.input_dims[1]*self.input_dims[2], self.num_filters])

            # Normalize weights (one or both dimensions)
            if self.normalize_weights in [0, 2]:
                wnorms = tf.sqrt(tf.reduce_sum(tf.square(kts), axis=0))
                kts = tf.divide(kts, tf.maximum(wnorms, 1e-6))
            if self.normalize_weights in [1, 2]:
                wnorms = tf.sqrt(tf.reduce_sum(tf.square(ksp), axis=0))
                ksp = tf.divide(ksp, tf.maximum(wnorms, 1e-6))

            weights_full = tf.transpose(tf.reshape(
                tf.matmul(tf.expand_dims(tf.transpose(ksp), 2),
                          tf.expand_dims(tf.transpose(kts), 1)),
                [self.num_filters, np.prod(self.input_dims)]))

            # Define computation
            if self.pos_constraint:
                pre = tf.add( tf.matmul(
                    inputs, tf.maximum(0.0, weights_full)), self.biases_var)
            else:
                pre = tf.add(
                    tf.matmul(inputs, weights_full), self.biases_var)

            if self.ei_mask_var is not None:
                post = tf.multiply(self.activation_func(pre), self.ei_mask_var)
            else:
                post = self.activation_func(pre)

            self.outputs = post

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END sepLayer._build_layer

    def write_layer_params(self, sess):
        """Write weights/biases in tf Variables to numpy arrays. Overloads function in layer
        in order to take care of normalization differences."""

        self.weights = sess.run(self.weights_var)

        if self.pos_constraint:
            self.weights = np.maximum(self.weights, 0)

        if self.normalize_weights in [0, 2]:
            kts = self.weights[range(self.input_dims[0]),:]
            wnorms = np.sqrt(np.sum(np.square(kts), axis=0))
            self.weights[range(self.input_dims[0]), :] = \
                np.divide(kts, np.maximum(wnorms, 1e-6))
            
        if self.normalize_weights in [1, 2]:
            ksp = self.weights[self.input_dims[0]:, :]
            wnorms = np.sqrt(np.sum(np.square(ksp), axis=0))
            self.weights[self.input_dims[0]:, :] = \
                np.divide(ksp, np.maximum(wnorms, 1e-6))

        self.biases = sess.run(self.biases_var)
    # END SepLayer.write_layer_params


class AddLayer(Layer):
    """Implementation of a simple additive layer that combines several input streams additively.
    This has a number of [output] units, and number of input streams, each which the exact same
    size as the number of output units. Each output unit then does a weighted sum over its matching
    inputs (with a weight for each input stream)

    """

    def __init__(
            self,
            scope=None,
            input_dims=None,  # this can be a list up to 3-dimensions
            output_dims=None,
            activation_func='relu',
            normalize_weights=0,
            reg_initializer=None,
            num_inh=0,
            pos_constraint=False,
            log_activations=False):
        """Constructor for sepLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int): dimensions of input data
            output_dims (int): dimensions of output data
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' |
                'elu' | 'quad'
            normalize_weights (int): type of normalization to apply to the
                weights. Default [0] is to normalize across the first dimension
                (time/filters), but '1' will normalize across spatial
                dimensions instead, and '2' will normalize both
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (bool, optional): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations
        """

        # check for required inputs
        if input_dims is None or output_dims is None:
            raise TypeError('Must specify input and output dimensions')

        num_outputs = np.prod( output_dims )
        num_input_streams = int(np.prod(input_dims) / num_outputs)

        # Input dims is just number of input streams
        input_dims = [num_input_streams, 1, 1]

        super(AddLayer, self).__init__(
                scope=scope,
                input_dims=input_dims,
                filter_dims=input_dims,
                output_dims=num_outputs,
                activation_func=activation_func,
                normalize_weights=normalize_weights,
                weights_initializer='zeros',
                biases_initializer='zeros',
                reg_initializer=reg_initializer,
                num_inh=num_inh,
                pos_constraint=pos_constraint,
                log_activations=log_activations)

        # Initialize all weights to 1, which is the default combination
        self.weights[:, :] = 1.0
        self.biases[:] = 1e-6

    # END AddLayer.__init__

    def build_graph(self, inputs, params_dict=None):
        """By definition, the inputs will be composed of a number of input streams, given by
        the first dimension of input_dims, and each stream will have the same number of inputs
        as the number of output units."""

        num_input_streams = self.input_dims[0]
        num_outputs = self.output_dims[1]
        # inputs will be NTx(num_input_streamsxnum_outputs)

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            if self.pos_constraint:
                ws = tf.maximum(0.0, self.weights_var)
            else:
                ws = self.weights_var

            if num_input_streams == 1:
                pre = tf.multiply( inputs, ws )
            else:

                if self.normalize_weights > 0:
                    wnorms = tf.sqrt(tf.reduce_sum(tf.square(self.weights_var), axis=0))
                    ws = tf.divide(self.weights_var, tf.maximum(wnorms, 1e-6))
                else:
                    ws = self.weights_var

                flattened_weights = tf.reshape(ws, [1, num_input_streams*num_outputs])
                # Define computation -- different from layer in that this is a broadcast-multiply
                # rather than  matmul
                pre = tf.multiply(inputs, flattened_weights)
                # Sum over input streams for given output
                pre = tf.reduce_sum(tf.reshape(pre, [-1, num_input_streams, num_outputs]), axis=1)

            pre = tf.add(pre, self.biases_var)

            if self.ei_mask_var is not None:
                post = tf.multiply(self.activation_func(pre), self.ei_mask_var)
            else:
                post = self.activation_func(pre)

            self.outputs = post

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END AddLayer._build_graph


class SpikeHistoryLayer(Layer):
    """Implementation of a simple additive layer that combines several input streams additively.
    This has a number of [output] units, and number of input streams, each which the exact same
    size as the number of output units. Each output unit then does a weighted sum over its matching
    inputs (with a weight for each input stream)

    """

    def __init__(
            self,
            scope=None,
            input_dims=None,  # this can be a list up to 3-dimensions
            output_dims=None,
            activation_func='relu',
            normalize_weights=0,
            reg_initializer=None,
            num_inh=0,
            pos_constraint=False,
            log_activations=False):
        """Constructor for sepLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int): dimensions of input data
            output_dims (int): dimensions of output data
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' |
                'elu' | 'quad'
            normalize_weights (int): type of normalization to apply to the
                weights. Default [0] is to normalize across the first dimension
                (time/filters), but '1' will normalize across spatial
                dimensions instead, and '2' will normalize both
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (bool, optional): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations
        """

        # check for required inputs
        if input_dims is None or output_dims is None:
            raise TypeError('Must specify input and output dimensions')
        filter_dims = input_dims[:]
        filter_dims[1] = 1

        super(SpikeHistoryLayer, self).__init__(
                scope=scope,
                input_dims=input_dims,
                filter_dims=filter_dims,
                output_dims=output_dims,
                activation_func=activation_func,
                normalize_weights=normalize_weights,
                weights_initializer='trunc_normal',
                biases_initializer='zeros',
                reg_initializer=reg_initializer,
                num_inh=num_inh,
                pos_constraint=pos_constraint,
                log_activations=log_activations)

        # Initialize all weights to be positive (and will be multiplied by -1
        self.biases[:] = 0

    # END SpikeHistorylayer.__init__

    def build_graph(self, inputs, params_dict=None):
        """By definition, the inputs will be composed of a number of input streams, given by
        the first dimension of input_dims, and each stream will have the same number of inputs
        as the number of output units."""

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            if self.pos_constraint:
                ws_flat = tf.reshape(tf.maximum(0.0, tf.transpose(self.weights_var)),
                                     [1, self.input_dims[0]*self.input_dims[1]])
            else:
                ws_flat = tf.reshape(tf.transpose(self.weights_var),
                                     [1, self.input_dims[0]*self.input_dims[1]])

            pre = tf.reduce_sum(tf.reshape(tf.multiply(inputs, ws_flat),
                                           [-1, self.input_dims[1], self.input_dims[0]]),
                                axis=2)

            # Dont put in any biases: pre = tf.add( pre, self.biases_var)
            if self.ei_mask_var is not None:
                post = tf.multiply(self.activation_func(pre), self.ei_mask_var)
            else:
                post = self.activation_func(pre)

            self.outputs = post

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END SpikeHistoryLayer._build_graph
