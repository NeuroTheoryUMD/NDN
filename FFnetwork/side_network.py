from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
#from .regularization import Regularization
from .ffnetwork import FFNetwork
#from .layer import sepLayer


class side_network(FFNetwork):
    """Implementation of siFFNetwork"""

    def __init__(self, scope=None, input_network=None, params_dict=None):
        """Constructor for Network class

        Args:
            scope (str): name scope for network
            inputs (tf Tensor or placeholder): input to network
            params_dict (dictionary with the following fields):
                SI-NETWORK SPECIFIC:
                -> first_filter_size: size of filters in first layer, if different than input size
                    DEFAULT = input size
                -> shift_spacing (int): convolutional "strides" to be passed back into conv2d
                    DEFAULT = 1
                -> binocular (boolean): currently doesn't work
                    DEFAULT = FALSE

                INHERITED FFNetwork PARAMS (see FFNetwork documentation):
                -> layer_sizes (list of ints)
                -> activation_funcs (str or list of strs, optional)
                -> weights_initializer (str or list of strs, optional)
                -> biases_initializer (str or list of strs, optional)
                -> reg_initializers (list of dicts)
                -> num_inh (None, int or list of ints, optional)
                -> pos_constraint (bool or list of bools, optional):
                -> log_activations (bool, optional)

        Raises:
            TypeError: If `scope` is not specified
            TypeError: If `inputs` is not specified
            TypeError: If `layer_sizes` is not specified
        """

        # Determine dimensions of input and pass into regular network initializer
        num_layers = len(input_network.layers)
        num_units = [0]*num_layers
        for nn in range(num_layers):
            num_units[nn] = np.prod(input_network.layers[nn].output_dims)

        super(side_network, self).__init__(
            scope=scope,
            input_dims = [num_layers, max(num_units), 1],
            params_dict=params_dict)

        self.num_units = num_units
    # END side_network.__init__

   def build_graph(self, input_network, params_dict=None):
        """Note this is different from other network build-graphs in that the whole
        network graph, rather than just a link to its output, so that it can be assembled here"""

        max_units = max(self.num_units)
        with tf.name_scope(self.scope):

            # Assemble network-inputs into the first layer
            inputs = []
            for input_nn in range(len(self.num_units)):
                inputs = tf.concat( 1, inputs, input_network.layers[input_nn].output )
                if max_units-self.num_units(input_nn) > 0:
                    inputs = tf.pad( input, [[max_units-self.num_units(input_nn),1]] )

            # Now standard graph-build (could just call the parent with inputs)
            for layer in range(self.num_layers):
                self.layers[layer].build_graph(inputs, params_dict)
                inputs = self.layers[layer].outputs
    # END side_network.build_graph


class side_layer(sepLayer):
    """Implementation of fully connected neural network layer

    Attributes:
        scope (str): name scope for variables and operations in layer
        num_inputs (int): number of inputs to layer
        num_outputs (int): number of outputs of layer
        outputs (tf Tensor): output of layer
        num_inh (int): number of inhibitory units in layer
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
        log (bool): use tf summary writers on layer output

    """

    def __init__(
            self,
            scope=None,
            input_network=None,
            output_dims=None,
            activation_func='relu',
            normalize_weights=False,
            weights_initializer='trunc_normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=False,
            log_activations=False):
        """Constructor for convLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int): dimension of input data
            num_outputs (int): dimension of output data
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' | 'elu' | 'quad'
            weights_initializer (str, optional): initializer for the weights
                ['trunc_normal'] | 'normal' | 'zeros'
            biases_initializer (str, optional): initializer for the biases
                'trunc_normal' | 'normal' | ['zeros']
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (bool, optional): True to constrain layer weights to be
                positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations

        Raises:
            TypeError: If `variable_scope` is not specified
            TypeError: If `inputs` is not specified
            TypeError: If `num_inputs` or `num_outputs` is not specified
            ValueError: If `num_inh` is greater than total number of units
            ValueError: If `activation_func` is not a valid string
            ValueError: If `weights_initializer` is not a valid string
            ValueError: If `biases_initializer` is not a valid string

        """

        # Process network input (which is an FFnetwork)
        num_layers = len(input_network.layers)
        num_units = [0]*num_layers
        for nn in range(num_layers):
            num_units[nn] = np.prod(input_network.layers[nn].output_dims)
        max_units = max(num_units)

        super(side_layer, self).__init__(
                scope=scope,
                input_dims = [num_layers, max_units, 1],
                output_dims = output_dims,
                activation_func=activation_func,
                normalize_weights=normalize_weights,
                weights_initializer=weights_initializer,
                biases_initializer=biases_initializer,
                reg_initializer=reg_initializer,
                num_inh=num_inh,
                pos_constraint=pos_constraint,
                log_activations=log_activations )

    # END side_layer.__init__
