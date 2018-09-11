"""Basic network-building tools"""

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from .layer import *
#from .layer import ConvLayer
#from .layer import SepLayer
#from .layer import ConvSepLayer
#from .layer import AddLayer
#from .layer import SpikeHistoryLayer
#from .layer import BiConvLayer

class FFNetwork(object):
    """Implementation of simple fully-connected feed-forward neural network. 
    These networks can be composed to create much more complex network 
    architectures using the NDN class.

    Attributes:
        input_dims (list of ints): three-element list containing the dimensions 
            of the input to the network, in the form 
            [num_lags, num_x_pix, num_y_pix]. If the input does not have 
            spatial or temporal structure, this should be [1, num_inputs, 1].
        scope (str): name scope for network
        num_layers (int): number of layers in network (not including input)
        layer_types (list of strs): a string for each layer in the network that 
            specifies its type.
            'normal' | 'sep' | 'conv' | 'convsep' | 'biconv' | 'add' | 'spike_history'
        layers (list of `Layer` objects): layers of network
        log (bool): use tf summary writers in layer activations

    """

    def __init__(self,
                 scope=None,
                 input_dims=None,
                 params_dict=None):
        """Constructor for FFNetwork class

        Args:
            scope (str): name scope for network
            input_dims (list of ints): three-element list containing the 
                dimensions of the input to the network, in the form 
                [num_lags, num_x_pix, num_y_pix]. If the input does not have 
                spatial or temporal structure, this should be 
                [1, num_inputs, 1]
            params_dict (dict): contains parameters about details of FFnetwork
            params_dict['layer_sizes'] (list of ints): list of layer sizes, 
                including input and output. All arguments (input size) can be 
                up to a 3-dimensional list. 
                REQUIRED (NO DEFAULT)
            params_dict['num_inh'] (int or list of ints): denotes number of 
                inhibitory units in each layer. This specifies the output of 
                that number of units multiplied by -1
                DEFAULT = 0 (and having any single value will be used for all 
                layers)
            params_dict['activation_funcs'] (str or list of strs, optional): 
                pointwise function for each layer; replicated if a single 
                element. 
                DEFAULT = 'relu'. See Layer class for other options.
            params_dict['pos_constraints'] (bool or list of bools, optional): 
                constrains all weights to be positive
                DEFAULTS = False.
            params_dict['reg_initializer'] (list): a list of dictionaries, one 
                for each layer. Within the dictionary, reg_type/vals as 
                key-value pairs.
                DEFAULT = None
            params_dict['weights_initializer'] (str or list of strs, optional): 
                initializer for the weights in each layer; replicated if a 
                single element.
                DEFAULT = 'trunc_normal'. See Layer class for other options.
            params_dict['biases_initializer'] (str or list of strs, optional): 
                initializer for the biases in each layer; replicated if a 
                single element.
                DEFAULT = 'zeros'. See Layer class for other options.
            params_dict['log_activations'] (bool, optional): True to use 
                tf.summary on layer activations
                DEFAULT = False

        Raises:
            TypeError: If `scope` is not specified
            TypeError: If `params_dict` is not specified
            TypeError: If `layer_sizes` is not specified in `params_dict`
            TypeError: If `input_dims` is `None` and is not contained in 
                `params_dict`
            TypeError: If 'layer_type` in `params_dict` is not a valid string
            ValueError: If `activation_funcs` in `params_dict` is not a 
                properly-sized list
            ValueError: If `weights_initializer` in `params_dict` is not a 
                properly-sized list
            ValueError: If `biases_initializer` in `params_dict` is not a 
                properly-sized list
            ValueError: If `biases_initializer` in `params_dict` is not a 
                properly-sized list
            ValueError: If `num_inh` in `params_dict` is not a 
                properly-sized list
            ValueError: If `pos_constraints` in `params_dict` is not a 
                properly-sized list
                
        """

        # check for required inputs
        if scope is None:
            raise TypeError('Must specify network scope')
        self.scope = scope

        if params_dict is None:
            raise TypeError('Must specify parameters dictionary.')

        if input_dims is None:
            input_dims = params_dict['input_dims']
            if params_dict['input_dims'] is None:
                raise TypeError('Must specify input dimensions.')
        # Format input dims (or check formatting)
        if not isinstance(input_dims, list):
            input_dims = [1, input_dims, 1]
        else:
            while len(input_dims) < 3:
                input_dims.append(1)
        self.input_dims = input_dims[:]

        # Check information in params_dict and set defaults
        if 'layer_sizes' not in params_dict:
            raise TypeError('Must specify layer_sizes.')

        self.num_layers = len(params_dict['layer_sizes'])
        self.layer_types = params_dict['layer_types']

        if 'activation_funcs' not in params_dict:
            params_dict['activation_funcs'] = 'relu'
        if type(params_dict['activation_funcs']) is not list:
            params_dict['activation_funcs'] = \
                [params_dict['activation_funcs']] * self.num_layers
        elif len(params_dict['activation_funcs']) != self.num_layers:
            raise ValueError('Invalid number of activation_funcs')

        if 'weights_initializers' not in params_dict:
            params_dict['weights_initializers'] = 'trunc_normal'
        if type(params_dict['weights_initializers']) is not list:
            params_dict['weights_initializers'] = \
                [params_dict['weights_initializers']] * self.num_layers
        elif len(params_dict['weights_initializers']) != self.num_layers:
            raise ValueError('Invalid number of weights_initializer')

        if 'biases_initializers' not in params_dict:
            params_dict['biases_initializers'] = 'zeros'
        if type(params_dict['biases_initializers']) is not list:
            params_dict['biases_initializers'] = \
                [params_dict['biases_initializers']] * self.num_layers
        elif len(params_dict['biases_initializers']) != self.num_layers:
            raise ValueError('Invalid number of biases_initializer')

        if 'reg_initializers' not in params_dict:
            params_dict['reg_initializers'] = [None] * self.num_layers

        if 'num_inh' not in params_dict:
            params_dict['num_inh'] = 0
        if type(params_dict['num_inh']) is not list:
            params_dict['num_inh'] = [params_dict['num_inh']] * self.num_layers
        elif len(params_dict['num_inh']) != self.num_layers:
            raise ValueError('Invalid number of num_inh')

        if 'pos_constraints' not in params_dict:
            params_dict['pos_constraints'] = False
        if type(params_dict['pos_constraints']) is not list:
            params_dict['pos_constraints'] = \
                [params_dict['pos_constraints']] * self.num_layers
        elif len(params_dict['pos_constraints']) != self.num_layers:
            raise ValueError('Invalid number of pos_con')

        if 'log_activations' not in params_dict:
            params_dict['log_activations'] = False

        # Define network
        with tf.name_scope(self.scope):
            self._define_network(params_dict)

        if params_dict['log_activations']:
            self.log = True
        else:
            self.log = False
    # END FFNetwork.__init__

    def _define_network(self, network_params):

        layer_sizes = [self.input_dims] + network_params['layer_sizes']
        self.layers = []
        #print(self.scope, layer_sizes)

        for nn in range(self.num_layers):

            if self.layer_types[nn] == 'normal':

                self.layers.append(Layer(
                    scope='layer_%i' % nn,
                    input_dims=layer_sizes[nn],
                    output_dims=layer_sizes[nn+1],
                    activation_func=network_params['activation_funcs'][nn],
                    normalize_weights=network_params['normalize_weights'][nn],
                    weights_initializer=network_params['weights_initializers'][nn],
                    biases_initializer=network_params['biases_initializers'][nn],
                    reg_initializer=network_params['reg_initializers'][nn],
                    num_inh=network_params['num_inh'][nn],
                    pos_constraint=network_params['pos_constraints'][nn],
                    log_activations=network_params['log_activations']))

            elif self.layer_types[nn] == 'sep':

                self.layers.append(SepLayer(
                    scope='sep_layer_%i' % nn,
                    input_dims=layer_sizes[nn],
                    output_dims=layer_sizes[nn+1],
                    activation_func=network_params['activation_funcs'][nn],
                    normalize_weights=network_params['normalize_weights'][nn],
                    weights_initializer=network_params['weights_initializers'][nn],
                    biases_initializer=network_params['biases_initializers'][nn],
                    reg_initializer=network_params['reg_initializers'][nn],
                    num_inh=network_params['num_inh'][nn],
                    pos_constraint=network_params['pos_constraints'][nn],
                    log_activations=network_params['log_activations']))

            elif self.layer_types[nn] == 'add':

                self.layers.append(AddLayer(
                    scope='add_layer_%i' % nn,
                    input_dims=layer_sizes[nn],
                    output_dims=layer_sizes[nn+1],
                    activation_func=network_params['activation_funcs'][nn],
                    normalize_weights=network_params['normalize_weights'][nn],
                    reg_initializer=network_params['reg_initializers'][nn],
                    num_inh=network_params['num_inh'][nn],
                    pos_constraint=network_params['pos_constraints'][nn],
                    log_activations=network_params['log_activations']))

            elif self.layer_types[nn] == 'spike_history':

                self.layers.append(SpikeHistoryLayer(
                    scope='spike_history_layer_%i' % nn,
                    input_dims=layer_sizes[nn],
                    output_dims=layer_sizes[nn+1],
                    activation_func=network_params['activation_funcs'][nn],
                    normalize_weights=network_params['normalize_weights'][nn],
                    reg_initializer=network_params['reg_initializers'][nn],
                    num_inh=network_params['num_inh'][nn],
                    pos_constraint=network_params['pos_constraints'][nn],
                    log_activations=network_params['log_activations']))

            elif self.layer_types[nn] == 'conv':

                if network_params['conv_filter_widths'][nn] is None:
                    conv_filter_size = layer_sizes[nn]
                else:
                    conv_filter_size = [
                        layer_sizes[nn][0],
                        network_params['conv_filter_widths'][nn], 1]
                    if layer_sizes[nn][2] > 1:
                        conv_filter_size[2] = \
                            network_params['conv_filter_widths'][nn]

                self.layers.append(ConvLayer(
                    scope='conv_layer_%i' % nn,
                    input_dims=layer_sizes[nn],
                    num_filters=layer_sizes[nn+1],
                    filter_dims=conv_filter_size,
                    shift_spacing=network_params['shift_spacing'][nn],
                    activation_func=network_params['activation_funcs'][nn],
                    normalize_weights=network_params['normalize_weights'][nn],
                    weights_initializer=network_params['weights_initializers'][nn],
                    biases_initializer=network_params['biases_initializers'][nn],
                    reg_initializer=network_params['reg_initializers'][nn],
                    num_inh=network_params['num_inh'][nn],
                    pos_constraint=network_params['pos_constraints'][nn],
                    log_activations=network_params['log_activations']))

                # Modify output size to take into account shifts
                if nn < self.num_layers:
                    layer_sizes[nn+1] = self.layers[nn].output_dims

            elif self.layer_types[nn] == 'convsep':

                if network_params['conv_filter_widths'][nn] is None:
                    conv_filter_size = layer_sizes[nn]
                else:
                    conv_filter_size = [
                        layer_sizes[nn][0],
                        network_params['conv_filter_widths'][nn], 1]
                    if layer_sizes[nn][2] > 1:
                        conv_filter_size[2] = \
                            network_params['conv_filter_widths'][nn]

                self.layers.append(ConvSepLayer(
                    scope='sepconv_layer_%i' % nn,
                    input_dims=layer_sizes[nn],
                    num_filters=layer_sizes[nn+1],
                    filter_dims=conv_filter_size,
                    shift_spacing=network_params['shift_spacing'][nn],
                    activation_func=network_params['activation_funcs'][nn],
                    normalize_weights=network_params['normalize_weights'][nn],
                    weights_initializer=network_params['weights_initializers'][nn],
                    biases_initializer=network_params['biases_initializers'][nn],
                    reg_initializer=network_params['reg_initializers'][nn],
                    num_inh=network_params['num_inh'][nn],
                    pos_constraint=network_params['pos_constraints'][nn],
                    log_activations=network_params['log_activations']))

                # Modify output size to take into account shifts
                if nn < self.num_layers:
                    layer_sizes[nn+1] = self.layers[nn].output_dims

            elif self.layer_types[nn] == 'biconv':

                if network_params['conv_filter_widths'][nn] is None:
                    conv_filter_size = layer_sizes[nn]
                else:
                    conv_filter_size = [
                        layer_sizes[nn][0],
                        network_params['conv_filter_widths'][nn], 1]
                    if layer_sizes[nn][2] > 1:
                        conv_filter_size[2] = \
                            network_params['conv_filter_widths'][nn]

                self.layers.append(BiConvLayer(
                    scope='conv_layer_%i' % nn,
                    input_dims=layer_sizes[nn],
                    num_filters=layer_sizes[nn+1],
                    filter_dims=conv_filter_size,
                    shift_spacing=network_params['shift_spacing'][nn],
                    activation_func=network_params['activation_funcs'][nn],
                    normalize_weights=network_params['normalize_weights'][nn],
                    weights_initializer=network_params['weights_initializers'][nn],
                    biases_initializer=network_params['biases_initializers'][nn],
                    reg_initializer=network_params['reg_initializers'][nn],
                    num_inh=network_params['num_inh'][nn],
                    pos_constraint=network_params['pos_constraints'][nn],
                    log_activations=network_params['log_activations']))

                # Modify output size to take into account shifts
                if nn < self.num_layers:
                    layer_sizes[nn+1] = self.layers[nn].output_dims

            else:
                raise TypeError('Layer type %i not defined.' % nn)

    # END FFNetwork._define_network

    def build_fit_variable_list(self, fit_parameter_list):
        """Makes a list of variables from this network that will be fit given 
        the fit_parameter_list"""

        var_list = []
        for layer in range(self.num_layers):
            if fit_parameter_list[layer]['weights']:
                var_list.append(self.layers[layer].weights_var)
            if fit_parameter_list[layer]['biases']:
                var_list.append(self.layers[layer].biases_var)
        return var_list
    # END FFNetwork.build_fit_variable_list

    def build_graph(self, inputs, params_dict=None):
        """Build tensorflow graph for this network"""

        with tf.name_scope(self.scope):
            for layer in range(self.num_layers):
                self.layers[layer].build_graph(inputs, params_dict)
                inputs = self.layers[layer].outputs
    # END FFNetwork._build_graph

    def assign_model_params(self, sess):
        """Read weights/biases in numpy arrays into tf Variables"""
        with tf.name_scope(self.scope):
            for layer in range(self.num_layers):
                self.layers[layer].assign_layer_params(sess)

    def write_model_params(self, sess):
        """Write weights/biases in tf Variables to numpy arrays"""
        for layer in range(self.num_layers):
            self.layers[layer].write_layer_params(sess)

    def assign_reg_vals(self, sess):
        """Update default tf Graph with new regularization penalties"""
        with tf.name_scope(self.scope):
            for layer in range(self.num_layers):
                self.layers[layer].assign_reg_vals(sess)

    def define_regularization_loss(self):
        """Build regularization loss portion of default tf graph"""
        with tf.name_scope(self.scope):
            # define regularization loss for each layer separately...
            reg_ops = [None for _ in range(self.num_layers)]
            for layer in range(self.num_layers):
                reg_ops[layer] = \
                    self.layers[layer].define_regularization_loss()
            # ...then sum over all layers
            reg_loss = tf.add_n(reg_ops)
        return reg_loss


class SideNetwork(FFNetwork):
    """Implementation of side network that takes input from multiple layers of
    other FFNetworks
    
    Attributes:
        num_units (int): number of output units of network
        
    """

    def __init__(self,
                 scope=None,
                 input_network_params=None,
                 params_dict=None):
        """Constructor for side_network class

        Args:
            scope (str): name scope for network
            input_network_params (dict): params_dict of `FFNetwork` that acts  
                as input to this network
            params_dict (dict): contains details about the network
            params_dict['first_filter_size'] (list of ints): size of filters 
                in first layer, if different than input size
                DEFAULT = input size
            params_dict['shift_spacing'] (int): convolutional "strides" to be 
                passed back into conv2d
                DEFAULT = 1
            params_dict['binocular'] (boolean): currently doesn't work
                DEFAULT = FALSE
            params_dict['layer_sizes'] (list of ints): see FFNetwork 
                documentation
            params_dict['activation_funcs'] (str or list of strs, optional): 
                see FFNetwork documentation
            params_dict['weights_initializer'] (str or list of strs, optional): 
                see FFNetwork documentation
            params_dict['biases_initializer'] (str or list of strs, optional): 
                see FFNetwork documentation
            params_dict['reg_initializers'] (list of dicts): 
                see FFNetwork documentation
            params_dict['num_inh'] (None, int or list of ints, optional): see 
                FFNetwork documentation
            params_dict['pos_constraint'] (bool or list of bools, optional): 
                see FFNetwork documentation
            params_dict['log_activations'] (bool, optional): see FFNetwork 
                documentation
                
        """

        # Determine dimensions of input and pass into regular network initializer
        input_layer_sizes = input_network_params['layer_sizes'][:]
        # Check if entire network is convolutional (then will have spatial input dims)
        all_convolutional = False
        nonconv_inputs = np.zeros(len(input_layer_sizes), dtype=int)
        if (input_network_params['layer_types'][0] == 'conv') or \
                (input_network_params['layer_types'][0] == 'biconv'):
            # then check that all are conv
            all_convolutional = True
            for nn in range(len(input_layer_sizes)):
                if input_network_params['layer_types'][nn] == 'conv' or \
                        (input_network_params['layer_types'][0] == 'biconv'):
                    nonconv_inputs[nn] = input_layer_sizes[nn]*input_network_params['input_dims'][1] *\
                                         input_network_params['input_dims'][2]
                else:
                    all_convolutional = False
                    nonconv_inputs[nn] = input_layer_sizes[nn]

        if all_convolutional:
            nx_ny = input_network_params['input_dims'][1:3]
            if input_network_params['layer_types'][0] == 'biconv':
                nx_ny[0] = int(nx_ny[0]/2)
                input_layer_sizes[0] = input_layer_sizes[0]*2
            input_dims = [max(input_layer_sizes)*len(input_layer_sizes), nx_ny[0], nx_ny[1]]
        else:
            nx_ny = [1, 1]
            input_dims = [len(input_layer_sizes), max(nonconv_inputs), 1]

        super(SideNetwork, self).__init__(
            scope=scope,
            input_dims=input_dims,
            params_dict=params_dict)

        self.num_space = nx_ny[0]*nx_ny[1]
        if all_convolutional:
            self.num_units = input_layer_sizes
        else:
            self.num_units = nonconv_inputs
    # END SideNetwork.__init__

    def build_graph(self, input_network, params_dict=None):
        """Note this is different from other network build-graphs in that the 
        whole network graph, rather than just a link to its output, so that it 
        can be assembled here"""

        max_units = max(self.num_units)
        num_layers = len(self.num_units)
        with tf.name_scope(self.scope):

            # Assemble network-inputs into the first layer
            for input_nn in range(num_layers):

                new_slice = input_network.layers[input_nn].outputs
                if max_units-self.num_units[input_nn] > 0:
                    layer_padding = tf.constant([
                        [0, 0],
                        [0, (max_units-self.num_units[input_nn])*self.num_space]])
                    new_slice = tf.pad(new_slice, layer_padding)

                if input_nn == 0:
                    inputs_raw = tf.expand_dims(new_slice, 2)
                else:
                    inputs_raw = tf.concat([inputs_raw, tf.expand_dims(new_slice, 2)], 2)

            # Need to put layer dimension with the filters as bottom dimension instead of top
            inputs = tf.reshape(inputs_raw, [-1, num_layers*max_units*self.num_space])

            # Now standard graph-build (could just call the parent with inputs)
            for layer in range(self.num_layers):
                self.layers[layer].build_graph(inputs, params_dict)
                inputs = self.layers[layer].outputs
    # END SideNetwork.build_graph
