"""Neural deep network"""

from __future__ import print_function
from __future__ import division
from copy import deepcopy

import numpy as np
import tensorflow as tf

from FFnetwork.ffnetwork import FFNetwork
from FFnetwork.ffnetwork import SideNetwork
from .network import Network
from NDNutils import concatenate_input_dims

from FFnetwork.tlayer import *
from FFnetwork.layer import *


class NDN(Network):
    """Tensorflow (tf) implementation of Neural Deep Network class

    Attributes:
        num_networks (int): number of FFNetworks in model
        network_list (list of `FFNetwork` objects): list containing all 
            FFNetworks in model
        num_input_streams (int): number of distinct input matrices processed by 
            the various FFNetworks in model
        ffnet_out (list of ints): indices into network_list of all FFNetworks 
            whose outputs are used in calculating the cost function
        input_sizes (list of lists): list of the form 
            [num_lags, num_x_pix, num_y_pix] that describes the input size for 
            each input stream
        output_sizes (list of ints): number of output units for each output 
            stream
        noise_dist (str): specifies the probability distribution used to define
            the cost function
            ['poisson'] | 'gaussian' | 'bernoulli'
        tf_seed (int): rng seed for both tensorflow and numpy, which allows 
            for reproducibly random initializations of parameters
        cost
        unit_cost
        cost_reg
        cost_penalized
        graph (tf.Graph object): tensorflow graph of the model
        saver (tf.train.Saver object): for saving and restoring models
        merge_summaries (tf.summary.merge_all op): for collecting all summaries
        init (tf.global_variables_initializer op): for initializing variables
        sess_config (tf.ConfigProto object): specifies configurations for 
            tensorflow session, such as GPU utilization

    Notes:
        One assumption is that the output of all FFnetworks -- whether or not
        they are multi-dimensional -- project down into one dimension when 
        input into a second network. As a result, concatenation of multiple 
        streams will always be in one dimension.
        
    """

    _allowed_noise_dists = ['gaussian', 'poisson', 'bernoulli']
    _allowed_network_types = ['normal', 'side']
    _allowed_layer_types = ['normal', 'conv', 'sep', 'convsep', 'add', 'biconv', 'spike_history']

    def __init__(
            self,
            network_list=None,
            noise_dist='poisson',
            ffnet_out=-1,
            input_dim_list=None,
            batch_size=None,
            tf_seed=0):
        """Constructor for network-NIM class

        Args:
            network_list (list of dicts): created using 
                `NDNutils.FFNetwork_params`
            noise_dist (str, optional): specifies the probability distribution  
                used to define the cost function
                ['poisson'] | 'gaussian' | 'bernoulli'
            ffnet_out (int or list of ints, optional): indices into 
                network_list that specifes which network outputs are used for 
                the cost function; defaults to last network in `network_list`
            input_dim_list (list of lists): list of the form 
                [num_lags, num_x_pix, num_y_pix] that describes the input size 
                for each input stream
            tf_seed (int)
            
        Raises:
            TypeError: If `network_list` is not specified
            ValueError: If `input_dim_list` does not match inputs specified for
                each FFNetwork in `network_list`
            ValueError: If any element of `ffnet_out` is larger than the length
                of `network_list`
            
        """

        if network_list is None:
            raise TypeError('Must specify network list.')

        # Call __init__() method of super class
        super(NDN, self).__init__()

        self.batch_size = batch_size

        # Set network_list
        if not isinstance(network_list, list):
            network_list = [network_list]
        self.num_networks = len(network_list)
        self.network_list = deepcopy(network_list)

        # Determine number of inputs
        measured_input_list = set()
        for nn in range(self.num_networks):
            if network_list[nn]['xstim_n'] is not None:
                measured_input_list |= set(network_list[nn]['xstim_n'])
        measured_input_list = list(measured_input_list)

        if input_dim_list is None:
            input_dim_list = [None] * len(measured_input_list)
        elif len(input_dim_list) < len(measured_input_list):
                ValueError('Something_wrong with inputs')

        self.num_input_streams = len(input_dim_list)

        if not isinstance(ffnet_out, list):
            ffnet_out = [ffnet_out]
        for nn in range(len(ffnet_out)):
            if ffnet_out[nn] > self.num_networks:
                ValueError('ffnet_out has values that are too big')

        self.ffnet_out = ffnet_out[:]
        self.input_sizes = input_dim_list[:]
        # list of output sizes (for Robs placeholders)
        self.output_sizes = [0] * len(ffnet_out)
        self.noise_dist = noise_dist
        self.tf_seed = tf_seed

        self._define_network()

        # set parameters for graph (constructed for each train)
        self.graph = None
        self.saver = None
        self.merge_summaries = None
        self.init = None
    # END NDN.__init__

    def _define_network(self):
        # Create the FFnetworks

        self.networks = []

        for nn in range(self.num_networks):
            # Tabulate network inputs. Note that multiple inputs assumed to be
            # combined along 2nd dim, and likewise 1-D outputs assumed to be
            # over 'space'
            input_dims_measured = None

            if self.network_list[nn]['ffnet_n'] is not None:
                ffnet_n = self.network_list[nn]['ffnet_n']
                for mm in ffnet_n:
                    assert mm <= self.num_networks, \
                        'Too many ffnetworks referenced.'
                    # print('network %i:' % nn, mm, input_dims_measured,
                    # self.networks[mm].layers[-1].output_dims )
                    input_dims_measured = concatenate_input_dims(
                        input_dims_measured,
                        self.networks[mm].layers[-1].output_dims)

            # Determine external inputs
            if self.network_list[nn]['xstim_n'] is not None:
                xstim_n = self.network_list[nn]['xstim_n']
                for mm in xstim_n:
                    # First see if input is not specified at NDN level
                    if self.input_sizes[mm] is None:
                        # then try to scrape from network_params
                        assert self.network_list[nn]['input_dims'] is not None, \
                            'External input size not defined.'
                        self.input_sizes[mm] = self.network_list[nn]['input_dims']
                    input_dims_measured = concatenate_input_dims(
                        input_dims_measured, self.input_sizes[mm])

            # Now specific/check input to this network
            if self.network_list[nn]['input_dims'] is None:
                if self.network_list[nn]['network_type'] != 'side':
                    self.network_list[nn]['input_dims'] = input_dims_measured
                # print('network %i:' % nn, input_dims_measured)
            else:
                # print('network %i:' % nn, network_list[nn]['input_dims'],
                # input_dims_measured )
                assert self.network_list[nn]['input_dims'] == \
                       list(input_dims_measured), 'Input_dims dont match.'

            # Build networks
            if self.network_list[nn]['network_type'] == 'side':
                assert len(self.network_list[nn]['ffnet_n']) == 1, \
                    'only one input to a side network'
                network_input_params = \
                    self.network_list[self.network_list[nn]['ffnet_n'][0]]
                self.networks.append(
                    SideNetwork(
                        scope='side_network_%i' % nn,
                        input_network_params=network_input_params,
                        params_dict=self.network_list[nn]))
            else:
                self.networks.append(
                    FFNetwork(
                        scope='network_%i' % nn,
                        params_dict=self.network_list[nn]))

        # Assemble outputs
        for nn in range(len(self.ffnet_out)):
            ffnet_n = self.ffnet_out[nn]
            self.output_sizes[nn] = \
                self.networks[ffnet_n].layers[-1].weights.shape[1]
    # END NDN._define_network

    def _build_graph(
            self,
            learning_alg='adam',
            opt_params=None,
            fit_variables=None):
        """NDN._build_graph"""

        # Take care of optimize parameters if necessary
        if opt_params is None:
            opt_params = self.optimizer_defaults({}, learning_alg)

        # overwrite unit_cost_norm with opt_params value (if specified)
        if opt_params['poisson_unit_norm'] is not None:
            self.poisson_unit_norm = opt_params['poisson_unit_norm']

        self.graph = tf.Graph()  # must be initialized before graph creation

        # for specifying device
        if opt_params['use_gpu']:
            self.sess_config = tf.ConfigProto(device_count={'GPU': 1})
        else:
            self.sess_config = tf.ConfigProto(device_count={'GPU': 0})

        # build model graph
        with self.graph.as_default():

            np.random.seed(self.tf_seed)
            tf.set_random_seed(self.tf_seed)

            # define pipeline for feeding data into model
            with tf.variable_scope('data'):
                self._initialize_data_pipeline()

            # Build network graph
            for nn in range(self.num_networks):

                if self.network_list[nn]['network_type'] == 'side':

                    # Specialized inputs to side-network
                    assert self.network_list[nn]['xstim_n'] is None, \
                        'Cannot have any external inputs into side network.'
                    assert len(self.network_list[nn]['ffnet_n']) == 1, \
                        'Can only have one network input into a side network.'
                    # Pass the entire network into the input of side network
                    input_network_n = self.network_list[nn]['ffnet_n'][0]
                    assert input_network_n < nn, \
                        'Must create network for side network first.'
                    input_cat = self.networks[input_network_n]

                else:   # assume normal network
                    # Assemble input streams -- implicitly along input axis 1
                    # (0 is T)
                    input_cat = None
                    if self.network_list[nn]['xstim_n'] is not None:
                        for ii in self.network_list[nn]['xstim_n']:
                            if input_cat is None:
                                input_cat = self.data_in_batch[ii]
                            else:
                                input_cat = tf.concat(
                                    (input_cat, self.data_in_batch[ii]),
                                    axis=1)
                    if self.network_list[nn]['ffnet_n'] is not None:
                        for ii in self.network_list[nn]['ffnet_n']:
                            if input_cat is None:
                                input_cat = \
                                    self.networks[ii].layers[-1].outputs
                            else:
                                input_cat = \
                                    tf.concat(
                                        (input_cat,
                                         self.networks[ii].layers[-1].outputs),
                                        axis=1)

                self.networks[nn].build_graph(input_cat, self.network_list[nn])

            # Define loss function
            with tf.variable_scope('loss'):
                self._define_loss()

            # Define optimization routine
            var_list = self._build_fit_variable_list(fit_variables)

            with tf.variable_scope('optimizer'):
                self._define_optimizer(
                    learning_alg=learning_alg,
                    opt_params=opt_params,
                    var_list=var_list)

            # add additional ops
            # for saving and restoring models (initialized after var creation)
            self.saver = tf.train.Saver()
            # collect all summaries into a single op
            self.merge_summaries = tf.summary.merge_all()
            # add variable initialization op to graph
            self.init = tf.global_variables_initializer()
    # END NDN._build_graph

    def _define_loss(self):
        """Loss function that will be used to optimize model parameters"""

        cost = []
        unit_cost = []
        for nn in range(len(self.ffnet_out)):
            data_out = self.data_out_batch[nn]
            if self.filter_data:
                # this will zero out predictions where there is no data,
                # matching Robs here
                pred = tf.multiply(
                    self.networks[self.ffnet_out[nn]].layers[-1].outputs,
                    self.data_filter_batch[nn])
            else:
                pred = self.networks[self.ffnet_out[nn]].layers[-1].outputs

            nt = tf.cast(tf.shape(pred)[0], tf.float32)
            # define cost function
            if self.noise_dist == 'gaussian':
                with tf.name_scope('gaussian_loss'):
                    cost.append(tf.nn.l2_loss(data_out - pred) / nt)
                    unit_cost.append(tf.reduce_mean(tf.square(data_out-pred), axis=0))

            elif self.noise_dist == 'poisson':
                with tf.name_scope('poisson_loss'):

                    if self.poisson_unit_norm is not None:
                        # normalize based on rate * time (number of spikes)
                        cost_norm = tf.multiply(self.poisson_unit_norm[nn], nt)
                    else:
                        cost_norm = nt

                    cost.append(-tf.reduce_sum(tf.divide(
                        tf.multiply(data_out, tf.log(self._log_min + pred)) - pred,
                        cost_norm)))

                    unit_cost.append(-tf.divide(
                        tf.reduce_sum(
                            tf.multiply(
                                data_out, tf.log(self._log_min + pred)) - pred, axis=0),
                        cost_norm))

            elif self.noise_dist == 'bernoulli':
                with tf.name_scope('bernoulli_loss'):
                    # Check per-cell normalization with cross-entropy
                    # cost_norm = tf.maximum(
                    #   tf.reduce_sum(data_out, axis=0), 1)
                    cost.append(tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=data_out, logits=pred)))
                    unit_cost.append(tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(
                                labels=data_out, logits=pred), axis=0))
            else:
                TypeError('Cost function not supported.')

        self.cost = tf.add_n(cost)
        self.unit_cost = unit_cost

        # add regularization penalties
        self.cost_reg = 0
        with tf.name_scope('regularization'):
            for nn in range(self.num_networks):
                self.cost_reg += self.networks[nn].define_regularization_loss()

        self.cost_penalized = tf.add(self.cost, self.cost_reg)

        # save summary of cost
        # with tf.variable_scope('summaries'):
        tf.summary.scalar('cost', self.cost)
        tf.summary.scalar('cost_penalized', self.cost_penalized)
        tf.summary.scalar('reg_pen', self.cost_reg)
    # END NDN._define_loss

    def _assign_model_params(self, sess):
        """Functions assigns parameter values to randomly initialized model"""
        with self.graph.as_default():
            for nn in range(self.num_networks):
                self.networks[nn].assign_model_params(sess)

    def _write_model_params(self, sess):
        """Pass write_model_params down to the multiple networks"""
        for nn in range(self.num_networks):
            self.networks[nn].write_model_params(sess)

    def _assign_reg_vals(self, sess):
        """Loops through all current regularization penalties and updates
        parameter values"""
        with self.graph.as_default():
            for nn in range(self.num_networks):
                self.networks[nn].assign_reg_vals(sess)

    def _build_fit_variable_list(self, fit_parameter_list):
        """Generates variable list to fit if argument is not none. 
        'fit_parameter_list' is generated by a """
        var_list = None
        if fit_parameter_list is not None:
            var_list = []
            for nn in range(self.num_networks):
                var_list += self.networks[nn].build_fit_variable_list(
                    fit_parameter_list[nn])
        return var_list
    # END _generate_variable_list

    def fit_variables(self, layers_to_skip=None, fit_biases=False):
        """Generates a list-of-lists-of-lists of correct format to specify all 
        the variables to fit, as an argument for network.train

        Args:
            layers_to_skip (list of lists, optional): list of layers to skip 
                for each network. If just single list, defaults to skipping 
                those layers in the first network
            fit_biases (bool or list of bools): specify whether or not to fit
                biases for each network
                DEFAULT: False
                
        """

        if layers_to_skip is None:
            layers_to_skip = [[]] * self.num_networks
        else:
            # Assume a single list is referring to the first network by default
            if not isinstance(layers_to_skip, list):
                layers_to_skip = [[layers_to_skip]]
            else:
                if not isinstance( layers_to_skip[0], list):
                    # then assume just meant for first network
                    layers_to_skip = [layers_to_skip]

        if not isinstance(fit_biases, list):
            fit_biases = [fit_biases]*self.num_networks
        # here assume a single list is referring to each network
        assert len(fit_biases) == self.num_networks, \
            'fit_biases must be a list over networks'
        for nn in range(self.num_networks):
            if not isinstance(fit_biases[nn], list):
                fit_biases[nn] = [fit_biases[nn]]*len(self.networks[nn].layers)
        fit_list = [[]]*self.num_networks
        for nn in range(self.num_networks):
            fit_list[nn] = [{}]*self.networks[nn].num_layers
            for layer in range(self.networks[nn].num_layers):
                fit_list[nn][layer] = {'weights': True, 'biases': False}
                fit_list[nn][layer]['biases'] = fit_biases[nn][layer]
                if nn < len(layers_to_skip):
                    if layer in layers_to_skip[nn]:
                        fit_list[nn][layer]['weights'] = False
                        fit_list[nn][layer]['biases'] = False
        return fit_list
        # END NDN.set_fit_variables


    def set_partial_fit(self, ffnet_target=0, layer_target=None, value=None):
        """ Assign partial_fit values

        Args:
            ffnet_target (int): which network
            layer_target (int): which layer
            value (0, 1, or None): partial_fit value to be assigned
                - 0 ---> fit only temporal part
                - 1 ---> fit only spatial part
                - anything else ---> fit everything
        """

        if value == 0:
            translation = 'only --temporal--'
        elif value == 1:
            translation = 'only --spatial--'
        else:
            translation = '--everything--'

        if type(self.networks[ffnet_target].layers[layer_target]) is SepLayer or ConvSepLayer:
            self.networks[ffnet_target].layers[layer_target].partial_fit = value
            self.networks[ffnet_target].layers[layer_target].reg.partial_fit = value

            print('....partial_fit value for --net%sL%s-- set to %s, %s will be fit'
                  % (ffnet_target, layer_target, value, translation))
        else:
            raise ValueError('partial fit should be used only with Seplayer families.')
    # END NDN.set_partial_fit

    def set_regularization(self, reg_type, reg_val, ffnet_target=0, layer_target=None):
        """Add or reassign regularization values

        Args:
            reg_type (str): see allowed_reg_types in regularization.py
            reg_val (int): corresponding regularization value
            ffnet_target (int): which network to assign regularization to
                DEFAULT: 0
            layer_target (int or list of ints): specifies which layers the
                current reg_type/reg_val pair is applied to 
                DEFAULT: all layers in ffnet_target

        """

        if layer_target is None:
            # set all layers
            for nn in range(self.num_networks):
                layer_target = range(self.networks[nn].num_layers)
        elif not isinstance(layer_target, list):
                layer_target = [layer_target]

        # set regularization at the layer level
        for layer in layer_target:
            self.networks[ffnet_target].layers[layer].set_regularization(
                reg_type, reg_val)
    # END set_regularization

    def get_LL(self, input_data, output_data, data_indxs=None,
               data_filters=None, use_gpu=False):
        """Get cost from loss function and regularization terms

        Args:
            input_data (time x input_dim numpy array): input to model
            output_data (time x output_dim numpy array): desired output of
                model
            data_indxs (numpy array, optional): indexes of data to use in
                calculating forward pass; if not supplied, all data is used
            data_filters (numpy array, optional):
            use_gpu (boolean, optional): whether or not to use GPU for
                computation (default=False)
        Returns:
            float: cost function evaluated on `input_data`

        """

        # check input
        if type(input_data) is not list:
            input_data = [input_data]
        if type(output_data) is not list:
            output_data = [output_data]
        if data_filters is not None:
            self.filter_data = True
            if type(data_filters) is not list:
                data_filters = [data_filters]
            assert len(data_filters) == len(output_data), \
                'Number of data filters must match output data.'
        self.num_examples = input_data[0].shape[0]
        for temp_data in input_data:
            if temp_data.shape[0] != self.num_examples:
                raise ValueError(
                    'Input data dims must match across input_data.')
        for nn, temp_data in enumerate(output_data):
            if temp_data.shape[0] != self.num_examples:
                raise ValueError('Output dim0 must match model values')
            if self.filter_data:
                assert data_filters[nn].shape == temp_data.shape, \
                    'data_filter sizes must match output_data'
        if data_indxs is None:
            data_indxs = np.arange(self.num_examples)

        # build datasets if using 'iterator' pipeline
        if self.data_pipe_type == 'iterator':
            dataset = self._build_dataset(
                input_data=input_data,
                output_data=output_data,
                data_filters=data_filters,
                indxs=data_indxs,
                training_dataset=False,
                batch_size=self.num_examples)
            # store info on dataset for buiding data pipeline
            self.dataset_types = dataset.output_types
            self.dataset_shapes = dataset.output_shapes
            # build iterator object to access elements from dataset
            iterator_tr = dataset.make_one_shot_iterator()

        # Potentially place graph operations on CPU
        if not use_gpu:
            #temp_config = tf.ConfigProto(device_count={'GPU': 0})
            with tf.device('/cpu:0'):
                self._build_graph()
        else:
            #temp_config = tf.ConfigProto(device_count={'GPU': 1})
            self._build_graph()

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:

            self._restore_params(sess, input_data, output_data, data_filters=data_filters)

            cost_tr = self._get_test_cost(
                sess=sess,
                input_data=input_data,
                output_data=output_data,
                data_filters=data_filters,
                test_indxs=data_indxs,
                test_batch_size=self.batch_size)

            #num_batches_tr = data_indxs.shape[0] // self.batch_size
            #cost_tr = 0
            #for batch_tr in range(num_batches_tr):
            #    batch_indxs_tr = data_indxs[
            #                     batch_tr * self.batch_size:(batch_tr + 1) * self.batch_size]
            #    if self.data_pipe_type == 'data_as_var':
            #        feed_dict = {self.indices: batch_indxs_tr}
            #    elif self.data_pipe_type == 'feed_dict':
            #        feed_dict = self._get_feed_dict(
            #            input_data=input_data,
            #            output_data=output_data,
            #            data_filters=data_filters,
            #            batch_indxs=batch_indxs_tr)
            #    elif self.data_pipe_type == 'iterator':
                    # get string handle of iterator
            #        iter_handle_tr = sess.run(iterator_tr.string_handle())
            #        feed_dict = {self.iterator_handle: iter_handle_tr}

            #    cost_tr += sess.run(self.cost, feed_dict=feed_dict)

            #cost_tr /= num_batches_tr

        return cost_tr
    # END get_LL

    def eval_models(self, input_data=None, output_data=None, data_indxs=None,
                    data_filters=None, nulladjusted=False, use_gpu=False):
        """Get cost for each output neuron without regularization terms

        Args:
            input_data (time x input_dim numpy array): input to model
            output_data (time x output_dim numpy array): desired output of
                model
            data_indxs (numpy array, optional): indexes of data to use in
                calculating forward pass; if not supplied, all data is used
            data_filters (numpy array, optional):
            nulladjusted (bool): subtracts the log-likelihood of a "null"
                model that has constant [mean] firing rate
            unit_norm (bool): if a Poisson noise-distribution, this will
                specify whether each LL will be normalized by the mean
                firing rate across all neurons (False) or by each neuron's
                own firing rate (True: default).

        Returns:
            numpy array: value of log-likelihood for each unit in model. For
                Poisson noise distributions, if not null-adjusted, this will
                return the *negative* log-likelihood. Null-adjusted will be
                positive if better than the null-model.
        """

        # check input
        if type(input_data) is not list:
            input_data = [input_data]
        if type(output_data) is not list:
            output_data = [output_data]
        if data_filters is not None:
            self.filter_data = True
            if type(data_filters) is not list:
                data_filters = [data_filters]
            assert len(data_filters) == len(output_data), \
                'Number of data filters must match output data.'
        self.num_examples = input_data[0].shape[0]
        for temp_data in input_data:
            if temp_data.shape[0] != self.num_examples:
                raise ValueError(
                    'Input data dims must match across input_data.')
        for nn, temp_data in enumerate(output_data):
            if temp_data.shape[0] != self.num_examples:
                raise ValueError('Output dim0 must match model values')
            if self.filter_data:
                assert data_filters[nn].shape == temp_data.shape, \
                    'data_filter sizes must match output_data'
        if data_indxs is None:
            data_indxs = np.arange(self.num_examples)

        # build datasets if using 'iterator' pipeline
        #if self.data_pipe_type == 'iterator':
        #    dataset = self._build_dataset(
        #        input_data=input_data,
        #        output_data=output_data,
        #        data_filters=data_filters,
        #        indxs=data_indxs,
        #        training_dataset=False,
        #        batch_size=self.num_examples)
        #    # store info on dataset for buiding data pipeline
        #    self.dataset_types = dataset.output_types
        #    self.dataset_shapes = dataset.output_shapes
        #    # build iterator object to access elements from dataset
        #    iterator = dataset.make_one_shot_iterator()

        # Place graph operations on CPU
        if not use_gpu:
            with tf.device('/cpu:0'):
                self._build_graph()
        else:
            self._build_graph()

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:

            self._restore_params(
                sess, input_data, output_data, data_filters=data_filters)

            if self.batch_size is not None:
                num_batches_test = data_indxs.shape[0] // self.batch_size

                for batch_test in range(num_batches_test):
                    batch_indxs_test = data_indxs[
                                       batch_test * self.batch_size:
                                       (batch_test + 1) * self.batch_size]
                    if self.data_pipe_type == 'data_as_var':
                        feed_dict = {self.indices: batch_indxs_test}
                    elif self.data_pipe_type == 'feed_dict':
                        feed_dict = self._get_feed_dict(
                            input_data=input_data,
                            output_data=output_data,
                            data_filters=data_filters,
                            batch_indxs=batch_indxs_test)
                    elif self.data_pipe_type == 'iterator':
                        feed_dict = {self.iterator_handle: data_indxs}

                    if batch_test == 0:
                        unit_cost = sess.run(self.unit_cost, feed_dict=feed_dict)
                    else:
                        unit_cost = np.add(unit_cost, sess.run(self.unit_cost, feed_dict=feed_dict))

                ll_neuron = np.divide(unit_cost, num_batches_test)

            else:  # no batch-size given
                if self.data_pipe_type == 'data_as_var':
                    feed_dict = {self.indices: data_indxs}
                elif self.data_pipe_type == 'feed_dict':
                    feed_dict = self._get_feed_dict(
                        input_data=input_data,
                        output_data=output_data,
                        data_filters=data_filters,
                        batch_indxs=data_indxs)
                elif self.data_pipe_type == 'iterator':
                    feed_dict = {self.iterator_handle: data_indxs}

                ll_neuron = sess.run(self.unit_cost, feed_dict=feed_dict)

            if nulladjusted:
                # note that ll_neuron is negative of the true log-likelihood,
                # but get_null_ll is not (so + is actually subtraction)
                for i, temp_data in enumerate(output_data):
                    ll_neuron[i] = -ll_neuron[i] - self.get_null_ll(temp_data[data_indxs, :])
            # note that this will only be correct for single output indices

        if len(ll_neuron) == 1:
            return ll_neuron[0]
        else:
            return ll_neuron
    # END get_LL_neuron

    def generate_prediction(self, input_data, data_indxs=None, use_gpu=False,
                            ffnet_target=-1, layer_target=-1):
        """Get cost for each output neuron without regularization terms

        Args:
            input_data (time x input_dim numpy array): input to model
            data_indxs (numpy array, optional): indexes of data to use in
                calculating forward pass; if not supplied, all data is used
            use_gpu (True or False): Obvious
            ffnet_target (int, optional): index into `network_list` that specifies
                which FFNetwork to generate the prediction from
            layer_target (int, optional): index into layers of network_list[ffnet_target]
                that specifies which layer to generate prediction from

        Returns:
            numpy array: pred values from network_list[ffnet_target].layers[layer]
                
        Raises:
            ValueError: If `layer` index is larger than number of layers in
                network_list[ffnet_target]

        """

        # check input
        if type(input_data) is not list:
            input_data = [input_data]
        self.num_examples = input_data[0].shape[0]
        for temp_data in input_data:
            if temp_data.shape[0] != self.num_examples:
                raise ValueError(
                    'Input data dims must match across input_data.')
        if data_indxs is None:
            data_indxs = np.arange(self.num_examples)
        if layer_target >= len(self.networks[ffnet_target].layers):
                ValueError('This layer does not exist.')
        if data_indxs is None:
            data_indxs = np.arange(self.num_examples)

        # change data_pipe_type to feed_dict
        #original_pipe_type = deepcopy(self.data_pipe_type)
        #self.data_pipe_type = 'data_as_var'

        # Generate fake_output data and take care of data-filtering, in case necessary
        self.filter_data = False
        num_outputs = len(self.ffnet_out)
        output_data = [None] * num_outputs
        for nn in range(num_outputs):
            output_data[nn] = np.zeros(
                [self.num_examples, self.networks[self.ffnet_out[nn]].layers[-1].weights.shape[1]],
                dtype='float32')

        if (self.batch_size is None) or (self.batch_size > data_indxs.shape[0]):
            batch_size_save = self.batch_size
            self.batch_size = data_indxs.shape[0]
        else:
            batch_size_save = None

        num_batches_test = data_indxs.shape[0] // self.batch_size

        # Place graph operations on CPU
        if not use_gpu:
            temp_config = tf.ConfigProto(device_count={'GPU': 0})
            with tf.device('/cpu:0'):
                self._build_graph()
        else:
            temp_config = tf.ConfigProto(device_count={'GPU': 1})
            self._build_graph()

        with tf.Session(graph=self.graph, config=temp_config) as sess:

            self._restore_params(sess, input_data, output_data)

            for batch_test in range(num_batches_test):

                batch_indxs_test = data_indxs[batch_test * self.batch_size:(batch_test + 1) * self.batch_size]
                feed_dict = {self.indices: batch_indxs_test}

                if batch_test == 0:
                    pred = sess.run(
                        self.networks[ffnet_target].layers[layer_target].outputs,
                        feed_dict=feed_dict)
                else:
                    pred = np.concatenate( (pred, sess.run(
                        self.networks[ffnet_target].layers[layer_target].outputs,
                        feed_dict=feed_dict)), axis=0)

            # Add last fraction (if batches did not include all data
            if pred.shape[0] < data_indxs.shape[0]:
                batch_indxs_test = data_indxs[num_batches_test * self.batch_size:data_indxs.shape[0]]
                feed_dict = {self.indices: batch_indxs_test}
                pred = np.concatenate((pred, sess.run(
                    self.networks[ffnet_target].layers[layer_target].outputs, feed_dict=feed_dict)), axis=0)

        # change the data_pipe_type to original
        # self.data_pipe_type = original_pipe_type
        if batch_size_save is not None:
            self.batch_size = batch_size_save

        return pred
    # END generate_prediction

    def get_reg_pen(self):
        """Return reg penalties in a dictionary"""

        reg_dict = {}
        if self.graph is None:
            self._build_graph()

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:

            # initialize all parameters randomly
            sess.run(self.init)

            # overwrite randomly initialized values of model with stored values
            self._assign_model_params(sess)

            # update regularization parameter values
            self._assign_reg_vals(sess)

            with tf.name_scope('get_reg_pen'):  # to keep the graph clean-ish
                for nn in range(self.num_networks):
                    for layer in range(self.networks[nn].num_layers):
                        reg_dict['net%iL%i' % (nn, layer)] = \
                            self.networks[nn].layers[layer].get_reg_pen(sess)

        return reg_dict

    def copy_model(self, tf_seed=0):
        """Makes an exact copy of model without further elaboration."""

        # Assemble network_list
        target = NDN(self.network_list, ffnet_out=self.ffnet_out,
                     noise_dist=self.noise_dist, tf_seed=tf_seed)

        target.poisson_unit_norm = self.poisson_unit_norm
        target.data_pipe_type = self.data_pipe_type
        target.batch_size = self.batch_size

        # Copy all the parameters
        for nn in range(self.num_networks):
            for ll in range(self.networks[nn].num_layers):
                target.networks[nn].layers[ll].weights = \
                    self.networks[nn].layers[ll ].weights.copy()
                target.networks[nn].layers[ll].biases = \
                    self.networks[nn].layers[ll].biases.copy()
                target.networks[nn].layers[ll].reg = \
                    self.networks[nn].layers[ll].reg.reg_copy()

        return target

    def matlab_export(self, filename):
        """Exports all weights and biases to matlab-readable file with given filename"""

        import scipy.io as sio
        matdata = {}
        for nn in range(self.num_networks):
            for ll in range(len(self.networks[nn].layers)):
                wstring = 'ws' + str(nn) + str(ll)
                matdata[wstring] = self.networks[nn].layers[ll].weights
                bstring = 'bs' + str(nn) + str(ll)
                matdata[bstring] = self.networks[nn].layers[ll].biases

        sio.savemat(filename, matdata)

    def initialize_output_layer_bias(self, robs):
        """Sets biases in output layer(w) to explain mean firing rate, using Robs"""

        if robs is not list:
            robs = [robs]
        for nn in range(len(self.ffnet_out)):
            frs = np.mean(robs[nn], axis=0, dtype='float32')
            assert len(frs) == len(
                self.networks[self.ffnet_out[nn]].layers[-1].biases[0]), \
                'Robs length wrong'
            self.networks[self.ffnet_out[nn]].layers[-1].biases = frs
    # END NDN.initialize_output_layer_bias

    def get_null_ll(self, robs):
        """Calculates null-model (constant firing rate) likelihood, given Robs 
        (which determines what firing rate for each cell)"""

        if self.noise_dist == 'gaussian':
            # In this case, LLnull is just var of data
            null_lls = np.var(robs, axis=0)

        elif self.noise_dist == 'poisson':
            rbars = np.mean(robs, axis=0)
            null_lls = np.log(rbars) - 1.0
        # elif self.noise_dist == 'bernoulli':
        else:
            null_lls = [0] * robs.shape[1]
            print('Not worked out yet')

        return null_lls
    # END NDN.get_null_ll

    def set_poisson_norm(self, data_out):
        """Calculates the average probability per bin to normalize the Poisson likelihood"""

        if type(data_out) is not list:
            data_out = [data_out]

        self.poisson_unit_norm = []
        for i, temp_data in enumerate(data_out):
            nc = self.network_list[self.ffnet_out[i]]['layer_sizes'][-1]
            assert nc == temp_data.shape[1], 'Output of network must match robs'

            self.poisson_unit_norm.append(np.maximum(np.mean(
                temp_data.astype('float32'), axis=0), 1e-8))

    # END NDN.set_poisson_norm
