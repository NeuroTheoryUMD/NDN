"""Neural deep network"""

from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from FFnetwork.ffnetwork import FFNetwork
from .network import Network
from NDNutils import expand_input_dims_to_3d
from NDNutils import concatenate_input_dims


class NDN(Network):
    """Tensorflow (tf) implementation of network-NIM class

    Attributes:
        network (FFNetwork object): feedforward network



    Notes:
        One assumption is that the output of all FFnetworks -- whether or not
        they are multi-dimensional -- project down into one dimension when input
        into a second network. As a result, concatination of multiple streams will
        always be in one dimension. If there is a single input into an FFnetwork,
    """

    _allowed_noise_dists = ['gaussian', 'poisson', 'bernoulli']
    _allowed_layer_types = ['normal', 'conv']

    def __init__(
            self,
            network_list = None,
            noise_dist='poisson',
            ffnet_out=-1,
            input_dim_list = None,
            tf_seed=0 ):

        """Constructor for network-NIM class

        Args:
            network_list: created using 'FFNetwork_params'
        """

        if network_list is None:
            raise TypeError('Must specify network list.')

        # Call __init__() method of super class
        super(NDN, self).__init__()

        # Set network_list
        if not isinstance( network_list, list ):
            network_list = [network_list]
        self.num_networks = len(network_list)
        self.network_list = network_list

        # Determine number of inputs
        measured_input_list = []
        for nn in range(self.num_networks):
            if network_list[nn]['xstim_n'] is not None:
                measured_input_list += network_list[nn]['xstim_n']
            measured_input_list = np.unique(measured_input_list)
        if input_dim_list is None:
            input_dim_list = [None]*len(measured_input_list)
        else:
            assert len(input_dim_list) >= len(measured_input_list), 'Something_wrong with inputs'
        self.num_input_streams = len(input_dim_list)

        if not isinstance(ffnet_out, list):
            ffnet_out = [ffnet_out]
        for nn in range(len(ffnet_out)):
            assert ffnet_out[nn] <= self.num_networks, 'ffnet_out has values that are too big'
        self.ffnet_out = ffnet_out
        self.input_sizes = input_dim_list
        self.output_size = [0]*len(ffnet_out)  # list of output sizes (for Robs placeholders)
        self.noise_dist = noise_dist
        self.tf_seed = tf_seed

        self._define_network(network_list)

        # set parameters for graph (constructed for each train)
        self.graph = None
        self.saver = None
        self.merge_summaries = None
        self.init = None

    # END networkNIM.__init__

    def _define_network(self, network_list ):
        # Create the FFnetworks

        self.networks = []

        for nn in range(self.num_networks):

            # Tabulate network inputs. Note that multiple inputs assumed to be combined along 2nd dim,
            # and likewise 1-D outputs assumed to be over 'space'
            input_dims_measured = None

            if network_list[nn]['ffnet_n'] is not None:
                ffnet_n = network_list[nn]['ffnet_n']
                for mm in ffnet_n:
                    assert ffnet_n[mm] <= self.num_networks, 'Too many ffnetworks referenced.'
                    #print('network %i:' % nn, mm, input_dims_measured, self.networks[mm].layers[-1].output_dims )
                    input_dims_measured = concatenate_input_dims(
                        input_dims_measured, self.networks[mm].layers[-1].output_dims)

            # Determine external inputs
            if network_list[nn]['xstim_n'] is not None:
                xstim_n = network_list[nn]['xstim_n']
                for mm in xstim_n:
                    # First see if input is not specified at NDN level
                    if self.input_sizes[mm] is None:
                        # then try to scrape from network_params
                        assert network_list[nn]['input_dims'] is not None, 'External input size not defined.'
                        self.input_sizes[mm] = network_list[nn]['input_dims']
                    input_dims_measured = concatenate_input_dims(
                        input_dims_measured, self.input_sizes[mm])

            # Now specific/check input to this network
            if network_list[nn]['input_dims'] is None:
                network_list[nn]['input_dims'] = input_dims_measured
                #print('network %i:' % nn, input_dims_measured)
            else:
                #print('network %i:' % nn, network_list[nn]['input_dims'], input_dims_measured )
                assert network_list[nn]['input_dims'] == list(input_dims_measured), 'Input_dims dont match.'

            # Build networks
            self.networks.append(
                FFNetwork(scope='network_%i' % nn, params_dict=network_list[nn]))

        # Assemble outputs
        for nn in range(len(self.ffnet_out)):
            ffnet_n = self.ffnet_out[nn]
            self.output_size[ffnet_n] = self.networks[ffnet_n].layers[-1].weights.shape[1]

        #print('num_inputs = ', self.input_sizes)
        #print('num_outputs = ', self.output_size)

    # END NDN._define_network

    def _build_graph(self, learning_alg='lbfgs', learning_rate=1e-3, use_gpu=False,
                     params_to_fit=None, data_filters=None ):

        # Check data_filters if it exists

        self.graph = tf.Graph()  # must be initialized before graph creation

        # for specifying device
        if use_gpu:
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

                if self.network_list[nn]['network_type'] is 'side':

                    # Specialized inputs to side-network
                    assert self.network_list[nn]['xstim_n'] is None, \
                        'Cannot have any external inputs into side network.'
                    assert len(self.network_list[nn]['ffnet_n']) == 1, \
                        'Can only have one network input into a side network.'
                    # Pass the entire network into the input of side network
                    input_network_n = self.network_list[nn]['ffnet_n'][0]
                    assert input_network_n < nn, 'Must create network for side network first.'
                    input_cat = self.networks[input_network_n]

                else: # assume normal network
                    # Assemble input streams -- implicitly along input axis 1 (0 is T)
                    input_cat = None
                    if self.network_list[nn]['xstim_n'] is not None:
                        for ii in self.network_list[nn]['xstim_n']:
                            if input_cat is None:
                                input_cat = self.data_in_batch[ii]
                            else:
                                input_cat = tf.concat( (input_cat, self.data_in_batch[ii]), axis=1 )
                    if self.network_list[nn]['ffnet_n'] is not None:
                        for ii in self.network_list[nn]['ffnet_n']:
                            if input_cat is None:
                                input_cat = self.networks[ii].layers[-1].outputs
                            else:
                                input_cat = tf.concat( (input_cat, self.networks[ii].layers[-1].outputs), axis=1 )

                self.networks[nn].build_graph(input_cat, self.network_list[nn])

            # Define loss function
            with tf.variable_scope('loss'):
                self._define_loss()

            # Define optimization routine
            var_list = self._build_fit_variable_list(params_to_fit)

            with tf.variable_scope('optimizer'):
                self._define_optimizer( learning_alg=learning_alg, learning_rate=learning_rate,
                                        var_list = var_list )

            # add additional ops
            # for saving and restoring models (initialized after var creation)
            self.saver = tf.train.Saver()
            # collect all summaries into a single op
            self.merge_summaries = tf.summary.merge_all()
            # add variable initialization op to graph
            self.init = tf.global_variables_initializer()

    def _define_loss(self):
        """Loss function that will be used to optimize model parameters"""

        cost = []
        self.unit_cost = []
        for nn in range(len(self.ffnet_out)):
            data_out = self.data_out_batch[nn]
            if self.filter_data:
                # this will zero out predictions where there is no data, matching Robs here
                pred = tf.multiply( self.networks[self.ffnet_out[nn]].layers[-1].outputs, self.data_filter_batch[nn] )
            else:
                pred = self.networks[self.ffnet_out[nn]].layers[-1].outputs

            # define cost function
            if self.noise_dist == 'gaussian':
                with tf.name_scope('gaussian_loss'):
                    # should variable 'cost' be defined here too?
                    cost.append( tf.nn.l2_loss(data_out - pred) / self.num_examples )
                    self.unit_cost = tf.concat(
                        [self.unit_cost, tf.reduce_mean(tf.square(data_out-pred), axis=0)], 0 )

            elif self.noise_dist == 'poisson':
                with tf.name_scope('poisson_loss'):
                    cost_norm = tf.maximum( tf.reduce_sum(data_out, axis=0), 1)
                    cost.append( -tf.reduce_sum( tf.divide(
                        tf.multiply(data_out,tf.log(self._log_min + pred)) - pred,
                        cost_norm ) ) )
                    self.unit_cost = tf.concat( [self.unit_cost, tf.divide( -tf.reduce_sum(
                        tf.multiply(data_out,tf.log(self._log_min + pred)) - pred, axis=0), cost_norm )], 0 )

            elif self.noise_dist == 'bernoulli':
                with tf.name_scope('bernoulli_loss'):
                    # Check per-cell normalization with cross-entropy
                    #cost_norm = tf.maximum( tf.reduce_sum(data_out, axis=0), 1)
                    cost.append( tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=data_out,logits=pred) ) )
                    self.unit_cost = tf.concat( [self.unit_cost, tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=data_out,logits=pred), axis=0 )], 0 )
                    #cost = tf.reduce_sum(self.unit_cost)
            else:
                print('Cost function not supported.')

        self.cost = tf.add_n(cost)

        # add regularization penalties
        self.cost_reg = 0
        with tf.name_scope('regularization'):
            for nn in range(self.num_networks):
                self.cost_reg += self.networks[nn].define_regularization_loss()

        self.cost_penalized = tf.add(self.cost, self.cost_reg)

        # save summary of cost
        with tf.variable_scope('summaries'):
            tf.summary.scalar('cost', cost)
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

    def _build_fit_variable_list( self, fit_parameter_list ):
        """Generates variable list to fit if argument is not none. 'fit_parameter_list'
        is generated by a """
        var_list = None
        if fit_parameter_list is not None:
            var_list = []
            for nn in range(self.num_networks):
                var_list += self.networks[nn]._build_fit_variable_list( fit_parameter_list[nn] )
        return var_list
    # END NDN._generate_variable_list

    def variables_to_fit(self, layers_to_skip=None, fit_biases=False):
        """Generates a list-of-lists-of-lists of correct format to specify all the
        variables to fit, as an argument for network.train

        Inputs:
            layers_to_skip: [default=None] this should be a list-of-lists, specifying
                a list of layers to skip for each network. If just single list, will assume
                this is skipping layers in the first network
            fit_biases: [default=False] this can be a single boolean value or list of values
                if want networks to have different default-bias-fitting states
            """

        if layers_to_skip is None:
            layers_to_skip = []
        else:
            if not isinstance(layers_to_skip,list):
                layers_to_skip = [layers_to_skip]
        if isinstance(fit_biases,list):
            assert len(fit_biases) == self.num_networks, 'fit_biases list must match the number of networks.'
        else:
            fit_biases = [fit_biases]*self.num_networks

        fit_list = []*self.num_networks
        for nn in range(self.num_networks):
            fit_list[nn] = [{}]*self.networks[nn].num_layers
            for layer in range(self.networks[nn].num_layers):
                fit_list[nn][layer]['weights']=True
                fit_list[nn][layer]['biases']=fit_biases
                if nn <= len(layers_to_skip):
                    if layer in layers_to_skip[nn]:
                        fit_list[nn][layer]['weights'] = False
                        fit_list[nn][layer]['biases'] = False

        return fit_list
        # END NDN.set_fit_variables

    def set_regularization(self, reg_type, reg_val, ffnet_n=0, layer_target=None):
        """Add or reassign regularization values

        Args:
            reg_type (str): see allowed_reg_types in regularization.py
            reg_val (int): corresponding regularization value
            ffnet_n(int): which network to assign regularization to (default = 0)
            layer_target (int or list of ints): specifies which layers the
                current reg_type/reg_val pair is applied to (default all in ffnet_n)

        """

        if layer_target is None:
            # set all layers
            for nn in range(self.num_networks):
                layer_target = range(self.networks[ffnet_n].num_layers)
        elif not isinstance(layer_target,list):
                layer_target = [layer_target]

        # set regularization at the layer level
        for layer in layer_target:
            new_reg_type = self.networks[ffnet_n].layers[layer].set_regularization(
                reg_type, reg_val)
    # END set_regularization

    def get_LL(self, input_data, output_data, data_indxs=None):
        """Get cost from loss function and regularization terms

        Args:
            input_data (time x input_dim numpy array): input to model
            output_data (time x output_dim numpy array): desired output of
                model
            data_indxs (numpy array, optional): indexes of data to use in
                calculating forward pass; if not supplied, all data is used

        Returns:
            cost (float): value of model's cost function evaluated on previous
                model data or that used as input
            reg_pen (float): value of model's regularization penalty

        Raises:
            ValueError: If data_in/out time dims don't match

        """

        # check input
        if data_indxs is None:
            data_indxs = np.arange(self.num_examples)

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:
            self._restore_params(sess, input_data, output_data)

            cost = sess.run(self.cost,
                            feed_dict={self.indices: data_indxs})

            return cost

    # END get_LL

    def eval_models(self, input_data, output_data, data_indxs=None):
        """Get cost for each output neuron without regularization terms

        Args:
            input_data (time x input_dim numpy array): input to model
            output_data (time x output_dim numpy array): desired output of
                model
            data_indxs (numpy array, optional): indexes of data to use in
                calculating forward pass; if not supplied, all data is used

        Returns:
            cost (float): value of model's cost function evaluated on previous
                model data or that used as input
            reg_pen (float): value of model's regularization penalty

        Raises:
            ValueError: If data_in/out time dims don't match

        """

        assert self.graph is not None, 'Must fit model first.'
        # check input
        if type(input_data) is not list:
            input_data = [input_data]
        if type(output_data) is not list:
            output_data = [output_data]
        if data_indxs is None:
            data_indxs = np.arange(self.num_examples)

        # Take care of data-filtering, in case necessary
        data_filters = [None]*len(output_data)
        for nn in range(len(output_data)):
            data_filters[nn] = np.ones( output_data[nn].shape, dtype='float32' )

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:
            self._restore_params(sess, input_data, output_data, data_filters)
            LL_neuron = sess.run(self.unit_cost, feed_dict={self.indices: data_indxs})

            return LL_neuron

    # END get_LL_neuron

    def generate_prediction(self, input_data, data_indxs=None, ffnet_n=-1, layer=-1):

        assert self.graph is not None, 'Must fit model first.'
        # check input
        if layer >= 0:
            assert layer < len(self.networks[ffnet_n].layers), 'This layer does not exist.'

        if data_indxs is None:
            data_indxs = np.arange(self.num_examples)

        # Generate fake_output data
        output_data = np.zeros( [self.num_examples, self.networks[ffnet_n].layers[-1].weights.shape[1]],
                                dtype='float32')

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:

            self._restore_params(sess, input_data, output_data)
            pred = sess.run(self.networks[ffnet_n].layers[layer].outputs, feed_dict={self.indices: data_indxs})

            return pred

    # END NetworkNIM.generate_prediction

    def get_reg_pen(self):
        """Return reg penalties in a dictionary"""

        reg_dict = {}
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
                        reg_dict['net%iL%i' % nn, layer] = \
                            self.networks[nn].layers[layer].get_reg_pen(sess)

        return reg_dict

    # END get_reg_pen


    def copy_model(self, tf_seed=0):
        """For the moment, this just makes exact-copy without further elaboration."""

        # Assemble network_list
        target = NDN( self.network_list, ffnet_out=self.ffnet_out,
                          noise_dist=self.noise_dist, tf_seed=tf_seed )
        return target

    # END NDN.copy_model
