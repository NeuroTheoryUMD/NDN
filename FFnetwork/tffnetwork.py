"""Temporal FFNetwork"""

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

#from .layer import Layer
#from .layer import ConvLayer
#from .layer import SepLayer
#from .layer import ConvSepLayer
#from .layer import AddLayer
#from .layer import SpikeHistoryLayer
#from .layer import BiConvLayer
#from .tlayer import TLayer
#from .tlayer import CaTentLayer
#from .tlayer import NoRollCaTentLayer

from .ffnetwork import FFNetwork

from .layer import *
from .tlayer import *



class TFFNetwork(FFNetwork):
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
                 params_dict=None,
                 batch_size=None,
                 time_spread=None):
        """Constructor for TFFnetwork class"""

        self.batch_size = batch_size
        self.time_spread = time_spread

        super(TFFNetwork, self).__init__(
            scope=scope,
            input_dims=input_dims,
            params_dict=params_dict)
    # END TFFNetwork.__init

    def _define_network(self, network_params):

        layer_sizes = [self.input_dims] + network_params['layer_sizes']
        self.layers = []

        for nn in range(self.num_layers):
            if self.layer_types[nn] == 'normal':
                self.layers.append(Layer(
                    scope='layer_%i' % nn,
                    nlags=network_params['time_expand'][nn],
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
                    nlags=network_params['time_expand'][nn],
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
                    nlags=network_params['time_expand'][nn],
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
                    nlags=network_params['time_expand'][nn],
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
                    nlags=network_params['time_expand'][nn],
                    input_dims=layer_sizes[nn],
                    num_filters=layer_sizes[nn+1],
                    filter_dims=conv_filter_size,
                    stride=network_params['stride'][nn],
                    dilation=network_params['dilation'][nn],
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

            elif self.layer_types[nn] == 'conv_xy':
                if network_params['conv_filter_widths'][nn] is None:
                    conv_filter_size = layer_sizes[nn]
                else:
                    conv_filter_size = [
                        layer_sizes[nn][0],
                        network_params['conv_filter_widths'][nn], 1]
                    if layer_sizes[nn][2] > 1:
                        conv_filter_size[2] = \
                            network_params['conv_filter_widths'][nn]

                self.layers.append(ConvXYLayer(
                    scope='conv_xy_layer_%i' % nn,
                    nlags=network_params['time_expand'][nn],
                    input_dims=layer_sizes[nn],
                    num_filters=layer_sizes[nn+1],
                    filter_dims=conv_filter_size,
                    xy_out=network_params['xy_out'][nn],
                    stride=network_params['stride'][nn],
                    dilation=network_params['dilation'][nn],
                    batch_size=self.batch_size,
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
                    scope='convsep_layer_%i' % nn,
                    nlags=network_params['time_expand'][nn],
                    input_dims=layer_sizes[nn],
                    num_filters=layer_sizes[nn+1],
                    filter_dims=conv_filter_size,
                    stride=network_params['stride'][nn],
                    dilation=network_params['dilation'][nn],
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

            elif self.layer_types[nn] == 'hadi_readout':
                self.layers.append(HadiReadoutLayer(
                    scope='conv_readout_layer_%i' % nn,
                    nlags=network_params['time_expand'][nn],
                    input_dims=layer_sizes[nn],
                    num_filters=layer_sizes[nn + 1],
                    xy_out=network_params['xy_out'][nn],
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
                    layer_sizes[nn + 1] = self.layers[nn].output_dims


            elif self.layer_types[nn] == 'gabor':
                if network_params['conv_filter_widths'][nn] is None:
                    conv_filter_size = layer_sizes[nn]
                else:
                    conv_filter_size = [
                        layer_sizes[nn][0],
                        network_params['conv_filter_widths'][nn], 1]
                    if layer_sizes[nn][2] > 1:
                        conv_filter_size[2] = \
                            network_params['conv_filter_widths'][nn]

                self.layers.append(GaborLayer(
                    scope='gabor_layer_%i' % nn,
                    nlags=network_params['time_expand'][nn],
                    input_dims=layer_sizes[nn],
                    num_filters=layer_sizes[nn+1],
                    filter_dims=conv_filter_size,
                    stride=network_params['stride'][nn],
                    dilation=network_params['dilation'][nn],
                    gabor_params_init=network_params['gabor_params_init'][nn],
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
                    scope='buiconv_layer_%i' % nn,
                    nlags=network_params['time_expand'][nn],
                    input_dims=layer_sizes[nn],
                    num_filters=layer_sizes[nn+1],
                    filter_dims=conv_filter_size,
                    stride=network_params['stride'][nn],
                    dilation=network_params['dilation'][nn],
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

            elif self.layer_types[nn] == 'temporal':
                self.layers.append(TLayer(
                    scope='temporal_layer_%i' % nn,
                    nlags=network_params['time_expand'][nn],
                    input_dims=layer_sizes[nn],
                    output_dims=layer_sizes[nn],
                    num_filters=layer_sizes[nn + 1],
                    filter_width=network_params['ca_tent_widths'][nn],
                    dilation=network_params['dilation'][nn],
                    activation_func=network_params['activation_funcs'][nn],
                    batch_size=self.batch_size,
                    normalize_weights=network_params['normalize_weights'][nn],
                    weights_initializer=network_params['weights_initializers'][nn],
                    biases_initializer=network_params['biases_initializers'][nn],
                    reg_initializer=network_params['reg_initializers'][nn],
                    num_inh=network_params['num_inh'][nn],
                    pos_constraint=network_params['pos_constraints'][nn],
                    log_activations=network_params['log_activations']))

                # Modify output size to take into account shifts
                if nn < self.num_layers:
                    layer_sizes[nn + 1] = self.layers[nn].output_dims

            elif self.layer_types[nn] == 'ca_tent':
                self.layers.append(CaTentLayer(
                    scope='ca_tent_layer_%i' % nn,
                    nlags=network_params['time_expand'][nn],
                    input_dims=layer_sizes[nn],
                    output_dims=layer_sizes[nn],
                    num_filters=layer_sizes[nn + 1],
                    filter_width=network_params['ca_tent_widths'][nn],
                    dilation=network_params['dilation'][nn],
                    activation_func=network_params['activation_funcs'][nn],
                    batch_size=self.batch_size,
                    normalize_weights=network_params['normalize_weights'][nn],
                    weights_initializer=network_params['weights_initializers'][nn],
                    biases_initializer=network_params['biases_initializers'][nn],
                    reg_initializer=network_params['reg_initializers'][nn],
                    num_inh=network_params['num_inh'][nn],
                    pos_constraint=network_params['pos_constraints'][nn],
                    log_activations=network_params['log_activations']))

                # Modify output size to take into account shifts
                if nn < self.num_layers:
                    layer_sizes[nn + 1] = self.layers[nn].output_dims

            elif self.layer_types[nn] == 'ca_tent_no_roll':
                self.layers.append(NoRollCaTentLayer(
                    scope='ca_tent_layer_%i' % nn,
                    input_dims=layer_sizes[nn],
                    nlags=network_params['time_expand'][nn],
                    output_dims=layer_sizes[nn],
                    num_filters=layer_sizes[nn + 1],
                    filter_width=network_params['ca_tent_widths'][nn],
                    batch_size=self.batch_size,
                    normalize_weights=network_params['normalize_weights'][nn],
                    weights_initializer=network_params['weights_initializers'][nn],
                    biases_initializer=network_params['biases_initializers'][nn],
                    reg_initializer=network_params['reg_initializers'][nn],
                    num_inh=network_params['num_inh'][nn],
                    pos_constraint=network_params['pos_constraints'][nn],
                    log_activations=network_params['log_activations']))

                # Modify output size to take into account shifts
                if nn < self.num_layers:
                    layer_sizes[nn + 1] = self.layers[nn].output_dims

            else:
                raise TypeError('Layer type %i not defined.' % nn)

    # END TFFNetwork._define_network

    def build_graph(self, inputs, params_dict=None, use_dropout=False):
        """Build tensorflow graph for this network"""

        with tf.name_scope(self.scope):
            for layer in range(self.num_layers):
                if self.layers[layer].nlags is not None:
                    inputs = time_expand(inputs=inputs,
                                         batch_sz=self.batch_size,
                                         nlags=self.layers[layer].nlags)
                    # no need to update input dims because it should be
                    # taken care of somewhere else
                    # self.layers[layer].input_dims = ???
                self.layers[layer].build_graph(inputs, params_dict, use_dropout=use_dropout)
                inputs = self.layers[layer].outputs
    # END TFFNetwork._build_graph



class TSideNetwork(TFFNetwork):
    """Implementation of side network that takes input from multiple layers of
    other FFNetworks

    Attributes:
        num_units (int): number of output units of network

    """

    def __init__(self,
                 scope=None,
                 input_network_params=None,
                 params_dict=None,
                 batch_size=None,
                 time_spread=None):
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
            params_dict['layer_sizes'] (list of ints): see FFNetwork documentation
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
            params_dict['log_activations'] (bool, optional): see FFNetwork documentation

        """

        _conv_types = ['conv', 'convsep', 'gabor', 'biconv', 'convLNL', 'conv_xy']
        isbinocular = False
        # Determine dimensions of input and pass into regular network initializer
        input_layer_sizes = input_network_params['layer_sizes'][:]
        # Check if entire network is convolutional (then will have spatial input dims)
        all_convolutional = False
        nonconv_inputs = np.zeros(len(input_layer_sizes), dtype=int)
        if input_network_params['layer_types'][0] in _conv_types:
            # then check that all are conv
            all_convolutional = True
            for nn in range(len(input_layer_sizes)):
                if input_network_params['layer_types'][nn] in _conv_types:
                    nonconv_inputs[nn] = input_layer_sizes[nn] * input_network_params['input_dims'][1] * \
                                         input_network_params['input_dims'][2]
                    if input_network_params['layer_types'][nn] == 'biconv':
                        isbinocular = True
                        # then twice as many outputs as filters
                        nonconv_inputs[nn] *= 2
                else:
                    all_convolutional = False
                    nonconv_inputs[nn] = input_layer_sizes[nn]
        else:
            nonconv_inputs = input_layer_sizes[:]

        if all_convolutional:
            nx_ny = input_network_params['input_dims'][1:]
            if isbinocular:
                nx_ny[0] = int(nx_ny[0] / 2)

                if input_network_params['layer_types'][0] == 'biconv':
                    input_layer_sizes[0] = input_layer_sizes[0] * 2
                elif input_network_params['layer_types'][1] == 'biconv':
                    input_layer_sizes[0] = input_layer_sizes[0] * 2
                    input_layer_sizes[1] = input_layer_sizes[1] * 2
            # input_dims = [max(input_layer_sizes)*len(input_layer_sizes), nx_ny[0], nx_ny[1]]

            input_dims = [np.sum(input_layer_sizes), nx_ny[0], nx_ny[1]]
        else:
            nx_ny = [1, 1]
            # input_dims = [len(input_layer_sizes), max(nonconv_inputs), 1]
            input_dims = [np.sum(nonconv_inputs), 1, 1]

        super(TSideNetwork, self).__init__(
            scope=scope,
            input_dims=input_dims,
            params_dict=params_dict,
            batch_size=batch_size,
            time_spread=time_spread)

        self.num_space = nx_ny[0] * nx_ny[1]
        if all_convolutional:
            self.num_units = input_layer_sizes
        else:
            self.num_units = nonconv_inputs

        # Set up potential side_network regularization (in first layer)
        self.layers[0].reg.scaffold_setup(self.num_units)

    # END TSideNetwork.__init__

    def build_graph(self, input_network, params_dict=None, use_dropout=False):
        """Note this is different from other network build-graphs in that the
        whole network graph, rather than just a link to its output, so that it
        can be assembled here"""

        num_layers = len(self.num_units)
        with tf.name_scope(self.scope):

            # Assemble network-inputs into the first layer
            for input_nn in range(num_layers):

                if (self.num_space == 1) or \
                        self.num_space == np.prod(input_network.layers[input_nn].output_dims[1:]):
                    new_slice = tf.reshape(input_network.layers[input_nn].outputs,
                                           [-1, self.num_space, self.num_units[input_nn]])
                else:  # spatial positions converted to different filters (binocular)
                    native_space = np.prod(input_network.layers[input_nn].output_dims[1:])
                    native_filters = input_network.layers[input_nn].output_dims[0]
                    tmp = tf.reshape(input_network.layers[input_nn].outputs,
                                     [-1, 1, native_space, native_filters])
                    # Reslice into correct spatial arrangement
                    left_post = tf.slice(tmp, [0, 0, 0, 0], [-1, -1, self.num_space, -1])
                    right_post = tf.slice(tmp, [0, 0, self.num_space, 0],
                                          [-1, -1, self.num_space, -1])

                    new_slice = tf.reshape(tf.concat([left_post, right_post], axis=3),
                                           [-1, self.num_space, self.num_units[input_nn]])

                if input_nn == 0:
                    inputs_raw = new_slice
                else:
                    inputs_raw = tf.concat([inputs_raw, new_slice], 2)

            # Need to put layer dimension with the filters as bottom dimension instead of top
            inputs = tf.reshape(inputs_raw, [-1, np.sum(self.num_units) * self.num_space])
            # inputs = tf.reshape(inputs_raw, [-1, num_layers*max_units*self.num_space])

            # Now standard graph-build (could just call the parent with inputs)
            for layer in range(self.num_layers):
                self.layers[layer].build_graph(inputs, params_dict, use_dropout=use_dropout)
                inputs = self.layers[layer].outputs
    # END TSideNetwork.build_graph


def get_tmat(batch_sz, nlags):
    """
    :param nlags:
    :param batch_sz:
    :return:
    """

    m = np.zeros((batch_sz, nlags, batch_sz))

    for lag in range(nlags):
        m[:, lag, :] = np.eye(batch_sz, k=-lag)

    return m


def time_expand(inputs, batch_sz, nlags):

    with tf.name_scope('time_expand'):

        tmat = get_tmat(batch_sz=batch_sz, nlags=nlags)
        tmat = tf.constant(tmat, dtype=tf.float32, name='tmat')

        expanded_inputs = tf.tensordot(tmat, inputs, axes=[2, 0])
        expanded_inputs_tr = tf.transpose(expanded_inputs, [0, 2, 1])
        expanded_inputs = tf.reshape(expanded_inputs_tr, (batch_sz, -1))

    return expanded_inputs

# define readout_computation here
