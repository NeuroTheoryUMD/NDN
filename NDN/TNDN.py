"""Temporal Neural Deep Network"""

from __future__ import print_function
from __future__ import division
from copy import deepcopy

import os
import numpy as np
import tensorflow as tf
import warnings
import shutil

from .NDN import NDN
from FFnetwork.ffnetwork import FFNetwork
from FFnetwork.ffnetwork import SideNetwork
from FFnetwork.tffnetwork import TFFNetwork

from NDNutils import concatenate_input_dims

from FFnetwork.tlayer import *


class TNDN(NDN):

    def __init__(
            self,
            network_list=None,
            noise_dist='gaussian',
            ffnet_out=-1,
            input_dim_list=None,
            batch_size=None,
            time_spread=None,
            tf_seed=0):
        """Constructor for temporal-NDN class

        Args:
            network_list (list of dicts): created using
                `NDNutils.FFNetwork_params`
                THIS MUCH INCLUDE BATCH-SIZE INFORMATION
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

      #  self.batch_size = batch_size
        self.time_spread = time_spread

        if network_list is None:
            raise TypeError('Must specify network list.')

        if batch_size is None:
            raise TypeError('Must specify batch size.')

        if time_spread is None:
            raise TypeError('Must specify batch size.')

        # Call __init__() method of super class
        super(TNDN, self).__init__(
            network_list=network_list,
            noise_dist=noise_dist,
            ffnet_out=ffnet_out,
            input_dim_list=input_dim_list,
            batch_size=batch_size,
            tf_seed=tf_seed)

    # END TNDN.__init

    def _set_batch_size(self, new_batch_size):
        """
        :param new_size:
        :return:
        """

        # TNDN
        if hasattr(self, 'batch_size'):
            self.batch_size = new_batch_size

        # TFFNetworks, layers
        for nn in range(self.num_networks):
            if hasattr(self.networks[nn], 'batch_size'):
                self.networks[nn].batch_size = new_batch_size
            for mm in range(len(self.networks[nn].layers)):
                if hasattr(self.networks[nn].layers[mm], 'batch_size'):
                    self.networks[nn].layers[mm].batch_size = new_batch_size
    # END TNDN._set_batch_size

    def _set_time_spread(self, new_time_spread):
        """
        :param new_size:
        :return:
        """

        # TNDN
        if hasattr(self, 'time_spread'):
            self.time_spread = new_time_spread

        # TFFNetworks, layers
        for nn in range(self.num_networks):
            if hasattr(self.networks[nn], 'time_spread'):
                self.networks[nn].time_spread = new_time_spread
            for mm in range(len(self.networks[nn].layers)):
                if hasattr(self.networks[nn].layers[mm], 'time_spread'):
                    self.networks[nn].layers[mm].time_spread = new_time_spread
    # END TNDN._set_time_spread


    def _define_network(self):
        # This code clipped from NDN, where TFFNetworks has to be added

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
            elif self.network_list[nn]['network_type'] == 'temporal':
                self.networks.append(
                    TFFNetwork(
                        scope='temporal_network_%i' % nn,
                        params_dict=self.network_list[nn],
                        batch_size=self.batch_size,
                        time_spread=self.time_spread))
            else:
                self.networks.append(
                    FFNetwork(
                        scope='network_%i' % nn,
                        params_dict=self.network_list[nn]))

        # Assemble outputs
        for nn in range(len(self.ffnet_out)):
            ffnet_n = self.ffnet_out[nn]

            if type(self.networks[ffnet_n].layers[-1]) is CaTentLayer:
                self.output_sizes[nn] = \
                    self.networks[ffnet_n].layers[-1].output_dims[1]
            else:
                self.output_sizes[nn] = \
                    self.networks[ffnet_n].layers[-1].weights.shape[1]

    # END TNDN._define_network

    def _define_loss(self):
        """Loss function that will be used to optimize model parameters"""

        cost = []
        unit_cost = []
        # interval used to calculate costs: range(self.time_spread//2, self.batch_size - self.time_spread//2)
        for nn in range(len(self.ffnet_out)):
            data_out = tf.slice(self.data_out_batch[nn],
                                [self.time_spread, 0],
                                [self.batch_size - self.time_spread - self.time_spread//2, -1])
            if self.filter_data:
                # this will zero out predictions where there is no data,
                # matching Robs here
                pred = tf.multiply(
                    self.networks[self.ffnet_out[nn]].layers[-1].outputs,
                    self.data_filter_batch[nn])
            else:
                pred = self.networks[self.ffnet_out[nn]].layers[-1].outputs

            pred = tf.slice(pred,
                            [self.time_spread, 0],
                            [self.batch_size - self.time_spread - self.time_spread//2, -1])

            # effective_batch_size is self.batch_size - 3/2 * self.time_spread
            effective_batch_size = tf.constant(
                self.batch_size - self.time_spread - self.time_spread//2, dtype=tf.float32)

            # define cost function
            if self.noise_dist == 'gaussian':
                with tf.name_scope('gaussian_loss'):
                    cost.append(
                        tf.nn.l2_loss(data_out - pred) / effective_batch_size)
                    unit_cost.append(tf.reduce_mean(tf.square(data_out-pred), axis=0))

            elif self.noise_dist == 'poisson':
                with tf.name_scope('poisson_loss'):

                    if self.poisson_unit_norm is not None:
                        # normalize based on rate * time (number of spikes)
                        cost_norm = tf.multiply(self.poisson_unit_norm[nn], effective_batch_size)
                    else:
                        cost_norm = effective_batch_size

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

    def train(
            self,
            input_data=None,
            output_data=None,
            train_indxs=None,
            test_indxs=None,
            variable_list=None,
            data_filters=None,
            learning_alg='lbfgs',
            opt_params=None,
            output_dir=None):
        """Network training function

        Args:
            input_data (list): input to network; each element should be a
                time x input_dim numpy array
            output_data (list of matrices): desired output of network; each
                element should be a time x output_dim numpy array
            train_indxs (numpy array, optional): subset of data to use for
                training
            test_indxs (numpy array, optional): subset of data to use for
                testing; if available these are used when displaying updates,
                and are also the indices used for early stopping if enabled
            variable_list (list-of-lists, optional): default none
                Generated by 'variables_to_fit' (if not none) to reference which
                variables in the model to fit.
            data_filters (list of matrices): matrices as same size as
                output_data that zeros out predictions where data is absent
            learning_alg (str, optional): algorithm used for learning
                parameters.
                ['lbfgs'] | 'adam'
            opt_params: dictionary with optimizer-specific parameters; see
                network.optimizer_defaults method for valid key-value pairs and
                corresponding default values.
            output_dir (str, optional): absolute path for saving checkpoint
                files and summary files; must be present if either
                `epochs_ckpt` or `epochs_summary` values in `opt_params` is not
                `None`. If `output_dir` is not `None`, regardless of checkpoint
                or summary settings, the graph will automatically be saved.
                Must be present if early_stopping is desired to restore the
                best fit, otherwise it will restore the model at break point.

        Returns:
            int: number of total training epochs

        Raises:
            ValueError: If `input_data` and `output_data` don't share time dim
            ValueError: If data time dim doesn't match that specified in model
            ValueError: If `epochs_ckpt` value in `opt_params` is not `None`
                and `output_dir` is `None`
            ValueError: If `epochs_summary` in `opt_params` is not `None` and
                `output_dir` is `None`
            ValueError: If `early_stop` > 0 and `test_indxs` is 'None'

        """

        self.num_examples = 0
        self.filter_data = False

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

        # Check format of opt_params (and add some defaults)
        if opt_params is None:
            opt_params = {}
        opt_params = self.optimizer_defaults(opt_params, learning_alg)

        # update data pipeline type before building tensorflow graph
        self.data_pipe_type = opt_params['data_pipe_type']

        if train_indxs is None:
            train_indxs = np.arange(self.num_examples)

        # Check values entered
        if learning_alg is 'adam':
            if opt_params['epochs_ckpt'] is not None and output_dir is None:
                raise ValueError(
                    'output_dir must be specified to save model')
            if opt_params['epochs_summary'] is not None and output_dir is None:
                raise ValueError(
                    'output_dir must be specified to save summaries')
            if opt_params['early_stop'] > 0 and test_indxs is None:
                raise ValueError(
                    'test_indxs must be specified for early stopping')

        # build datasets if using 'iterator' pipeline
        if self.data_pipe_type is 'iterator':
            dataset_tr = self._build_dataset(
                input_data=input_data,
                output_data=output_data,
                data_filters=data_filters,
                indxs=train_indxs,
                training_dataset=True,
                batch_size=opt_params['batch_size'])
            # store info on dataset for buiding data pipeline
            self.dataset_types = dataset_tr.output_types
            self.dataset_shapes = dataset_tr.output_shapes
            if test_indxs is not None:
                dataset_test = self._build_dataset(
                    input_data=input_data,
                    output_data=output_data,
                    data_filters=data_filters,
                    indxs=test_indxs,
                    training_dataset=False,
                    batch_size=opt_params['batch_size'])
            else:
                dataset_test = None

        # Set Poisson_unit_norm if specified
        if opt_params['poisson_unit_norm'] is not None:
            self.poisson_unit_norm = opt_params['poisson_unit_norm']
        elif (self.noise_dist == 'poisson') and (self.poisson_unit_norm is None):
            self.set_poisson_norm(output_data)

        # Build graph: self.build_graph must be defined in child of network
        self._build_graph(
            learning_alg=learning_alg,
            opt_params=opt_params,
            variable_list=variable_list)

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:

            # handle output directories
            train_writer = None
            test_writer = None
            if output_dir is not None:

                # remake checkpoint directory
                if opt_params['epochs_ckpt'] is not None:
                    ckpts_dir = os.path.join(output_dir, 'ckpts')
                    if os.path.isdir(ckpts_dir):
                        tf.gfile.DeleteRecursively(ckpts_dir)
                    os.makedirs(ckpts_dir)

                # remake training summary directories
                summary_dir_train = os.path.join(
                    output_dir, 'summaries', 'train')
                if os.path.isdir(summary_dir_train):
                    tf.gfile.DeleteRecursively(summary_dir_train)
                os.makedirs(summary_dir_train)
                train_writer = tf.summary.FileWriter(
                    summary_dir_train, graph=sess.graph)

                # remake testing summary directories
                summary_dir_test = os.path.join(
                    output_dir, 'summaries', 'test')
                if test_indxs is not None:
                    if os.path.isdir(summary_dir_test):
                        tf.gfile.DeleteRecursively(summary_dir_test)
                    os.makedirs(summary_dir_test)
                    test_writer = tf.summary.FileWriter(
                        summary_dir_test, graph=sess.graph)

            # overwrite initialized values of network with stored values
            self._restore_params(sess, input_data, output_data, data_filters)

            if self.data_pipe_type is 'data_as_var':
                # select learning algorithm
                if learning_alg is 'adam':
                    epoch = self._train_adam(
                        sess=sess,
                        train_writer=train_writer,
                        test_writer=test_writer,
                        train_indxs=train_indxs,
                        test_indxs=test_indxs,
                        opt_params=opt_params,
                        output_dir=output_dir)
                elif learning_alg is 'lbfgs':
                    self.train_step.minimize(
                        sess, feed_dict={self.indices: train_indxs})
                    epoch = float('NaN')
                else:
                    raise ValueError('Invalid learning algorithm')

            elif self.data_pipe_type is 'feed_dict':
                # select learning algorithm
                if learning_alg is 'adam':
                    epoch = self._train_adam(
                        sess=sess,
                        train_writer=train_writer,
                        test_writer=test_writer,
                        train_indxs=train_indxs,
                        test_indxs=test_indxs,
                        input_data=input_data,
                        output_data=output_data,
                        data_filters=data_filters,
                        opt_params=opt_params,
                        output_dir=output_dir)

                elif learning_alg is 'lbfgs':
                    feed_dict = self._get_feed_dict(
                        input_data=input_data,
                        output_data=output_data,  # this line needed?
                        batch_indxs=train_indxs)

                    self.train_step.minimize(sess, feed_dict=feed_dict)
                    epoch = float('NaN')
                else:
                    raise ValueError('Invalid learning algorithm')

            elif self.data_pipe_type is 'iterator':
                # select learning algorithm
                if learning_alg is 'adam':
                    epoch = self._train_adam(
                        sess=sess,
                        train_writer=train_writer,
                        test_writer=test_writer,
                        train_indxs=train_indxs,
                        test_indxs=test_indxs,
                        dataset_tr=dataset_tr,
                        dataset_test=dataset_test,
                        opt_params=opt_params,
                        output_dir=output_dir)

                elif learning_alg is 'lbfgs':
                    raise ValueError(
                        'Use of iterator pipeline with lbfgs not supported')
                else:
                    raise ValueError('Invalid learning algorithm')

            # write out weights/biases to numpy arrays before session closes
            self._write_model_params(sess)

        return epoch
    # END TNDN.train

    def _train_adam(
            self,
            sess=None,
            train_writer=None,
            test_writer=None,
            train_indxs=None,
            test_indxs=None,
            input_data=None,
            output_data=None,
            data_filters=None,
            dataset_tr=None,
            dataset_test=None,
            opt_params=None,
            output_dir=None):
        """Training function for adam optimizer to clean up code in `train`"""

        epochs_training = opt_params['epochs_training']
        epochs_ckpt = opt_params['epochs_ckpt']
        epochs_summary = opt_params['epochs_summary']
        # Inherit batch size if relevant
    #    if opt_params['batch_size'] is not None:
    #        self.batch_size = opt_params['batch_size']

        # print batch size
        print('\n*** Train initiated, batch size = %s ***\n\n' % self.batch_size)

        if self.data_pipe_type != 'data_as_var':
            assert self.batch_size is not None, 'Need to assign batch_size to train.'

        if opt_params['early_stop'] > 0:
            prev_costs = np.multiply(np.ones(opt_params['early_stop']), float('NaN'))

        # get number of batches and their order for train indxs
        num_batches_tr = train_indxs.shape[0] // self.batch_size
        batch_order = np.arange(num_batches_tr)

        if opt_params['run_diagnostics']:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        # build iterator handles if using that input pipeline type
        if self.data_pipe_type == 'iterator':
            # build iterator object to access elements from dataset
            iterator_tr = dataset_tr.make_one_shot_iterator()
            # get string handle of iterator
            iter_handle_tr = sess.run(iterator_tr.string_handle())

            if test_indxs is not None:
                # build iterator object to access elements from dataset
                iterator_test = dataset_test.make_one_shot_iterator()
                # get string handle of iterator
                iter_handle_test = sess.run(iterator_test.string_handle())

        # used in early_stopping
        best_epoch = 0
        best_cost = float('Inf')
        chkpted = False

        # start training loop
        for epoch in range(epochs_training):

            # a random permutation of batch order
            batch_order_perm = np.random.permutation(batch_order)

            # pass through dataset once
            for ii in range(num_batches_tr):
                if (self.data_pipe_type == 'data_as_var') or (
                        self.data_pipe_type == 'feed_dict'):
                    # get training indices for this batch
                    batch_indxs = train_indxs[
                                  batch_order_perm[ii] * self.batch_size:
                                  (batch_order_perm[ii] + 1) * self.batch_size]

                # one step of optimization routine
                if self.data_pipe_type == 'data_as_var':
                    # get the feed_dict for batch_indxs
                    feed_dict = {self.indices: batch_indxs}
                elif self.data_pipe_type == 'feed_dict':
                    feed_dict = self._get_feed_dict(
                        input_data=input_data,
                        output_data=output_data,
                        data_filters=data_filters,
                        batch_indxs=batch_indxs)
                elif self.data_pipe_type == 'iterator':
                    feed_dict = {self.iterator_handle: iter_handle_tr}

                sess.run(self.train_step, feed_dict=feed_dict)

            # print training updates
            if opt_params['display'] is not None and \
                    (epoch % opt_params['display'] == opt_params['display'] - 1
                     or epoch == 0):

                cost_tr, cost_test, reg_pen = 0, 0, 0
                for batch_tr in range(num_batches_tr):
                    batch_indxs_tr = train_indxs[
                                     batch_tr * self.batch_size:
                                     (batch_tr + 1) * self.batch_size]
                    if self.data_pipe_type == 'data_as_var':
                        feed_dict = {self.indices: batch_indxs_tr}
                    elif self.data_pipe_type == 'feed_dict':
                        feed_dict = self._get_feed_dict(
                            input_data=input_data,
                            output_data=output_data,
                            data_filters=data_filters,
                            batch_indxs=batch_indxs_tr)
                    elif self.data_pipe_type == 'iterator':
                        feed_dict = {self.iterator_handle: iter_handle_tr}

                    cost_tr += sess.run(self.cost, feed_dict=feed_dict)
                    reg_pen += sess.run(self.cost_reg, feed_dict=feed_dict)
                cost_tr /= num_batches_tr
                reg_pen /= num_batches_tr

                if test_indxs is not None:
                    if self.data_pipe_type == 'data_as_var' or \
                            self.data_pipe_type == 'feed_dict':
                        cost_test = self._get_test_cost(
                            sess=sess,
                            input_data=input_data,
                            output_data=output_data,
                            data_filters=data_filters,
                            test_indxs=test_indxs,
                            test_batch_size=opt_params['batch_size_test'])
                    elif self.data_pipe_type == 'iterator':
                        cost_test = self._get_test_cost(
                            sess=sess,
                            input_data=input_data,
                            output_data=output_data,
                            data_filters=data_filters,
                            test_indxs=iter_handle_test,
                            test_batch_size=opt_params['batch_size_test'])

                # print additional testing info
                print('Epoch %04d:   train cost  =  %.2e ,   '
                      'test cost  =  %.2e ,   '
                      'reg pen  =  %.2e'
                      % (epoch, cost_tr / np.sum(self.output_sizes),
                         cost_test / np.sum(self.output_sizes),
                         reg_pen / np.sum(self.output_sizes)))

            # save model checkpoints
            if epochs_ckpt is not None and (
                    epoch % epochs_ckpt == epochs_ckpt - 1 or epoch == 0):
                save_file = os.path.join(
                    output_dir, 'ckpts',
                    str('epoch_%05g.ckpt' % epoch))
                self.checkpoint_model(sess, save_file)

            # save model summaries
            if epochs_summary is not None and \
                    (epoch % epochs_summary == epochs_summary - 1
                     or epoch == 0):

                # TODO: what to use with feed_dict?
                if opt_params['run_diagnostics']:
                    summary = sess.run(
                        self.merge_summaries,
                        feed_dict=feed_dict,
                        options=run_options,
                        run_metadata=run_metadata)
                    train_writer.add_run_metadata(
                        run_metadata, 'epoch_%d' % epoch)
                else:
                    summary = sess.run(
                        self.merge_summaries,
                        feed_dict=feed_dict)
                train_writer.add_summary(summary, epoch)
                train_writer.flush()

                if test_indxs is not None:
                    if opt_params['run_diagnostics']:
                        summary = sess.run(
                            self.merge_summaries,
                            feed_dict=feed_dict,
                            options=run_options,
                            run_metadata=run_metadata)
                        test_writer.add_run_metadata(
                            run_metadata, 'epoch_%d' % epoch)
                    else:
                        summary = sess.run(
                            self.merge_summaries,
                            feed_dict=feed_dict)
                    test_writer.add_summary(summary, epoch)
                    test_writer.flush()

            if opt_params['early_stop'] > 0:

                # if you want to suppress that useless warning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_before = np.nanmean(prev_costs)

                if self.data_pipe_type == 'data_as_var' or \
                        self.data_pipe_type == 'feed_dict':
                    cost_test = self._get_test_cost(
                        sess=sess,
                        input_data=input_data,
                        output_data=output_data,
                        data_filters=data_filters,
                        test_indxs=test_indxs,
                        test_batch_size=opt_params['batch_size_test'])
                elif self.data_pipe_type == 'iterator':
                    cost_test = self._get_test_cost(
                        sess=sess,
                        input_data=input_data,
                        output_data=output_data,
                        data_filters=data_filters,
                        test_indxs=iter_handle_test,
                        test_batch_size=opt_params['batch_size_test'])

                prev_costs = np.roll(prev_costs, 1)
                prev_costs[0] = cost_test

                mean_now = np.nanmean(prev_costs)

                delta = (mean_before - mean_now) / mean_before

                # to check and refine the condition on chkpting best model
                # print(epoch, delta, 'delta condition:', delta < 1e-4)

                if cost_test < best_cost:
                    # update best cost and the epoch that it happened at
                    best_cost = cost_test
                    best_epoch = epoch
                    # chkpt model if desired
                    if output_dir is not None:
                        if opt_params['early_stop_mode'] == 1:
                            save_file = os.path.join(output_dir,
                                                     'bstmods', 'best_model')
                            self.checkpoint_model(sess, save_file)
                            chkpted = True
                        elif opt_params['early_stop_mode'] == 2 and \
                                delta < 5e-5:
                            save_file = os.path.join(output_dir,
                                                     'bstmods', 'best_model')
                            self.checkpoint_model(sess, save_file)
                            chkpted = True

                if opt_params['early_stop_mode'] == 1:
                    if (epoch > opt_params['early_stop'] and
                            mean_now >= mean_before):  # or equivalently delta <= 0
                        print('\n*** early stop criteria met...'
                              'stopping train now...')
                        print('     ---> number of epochs used: %d,  '
                              'end cost: %04f' % (epoch, cost_test))
                        print('     ---> best epoch: %d,  '
                              'best cost: %04f\n' % (best_epoch, best_cost))
                        # restore saved variables into tf Variables
                        if output_dir is not None and chkpted and \
                                opt_params['early_stop_mode'] > 0:
                            # save_file exists only if chkpted is True
                            self.saver.restore(sess, save_file)
                            # delete files before break to clean up space
                            shutil.rmtree(os.path.join(output_dir, 'bstmods'),
                                          ignore_errors=True)
                        break
                else:
                    if mean_now >= mean_before:  # or equivalently delta <= 0
                        print('\n*** early stop criteria met...'
                              'stopping train now...')
                        print('     ---> number of epochs used: %d,  '
                              'end cost: %04f' % (epoch, cost_test))
                        print('     ---> best epoch: %d,  '
                              'best cost: %04f\n' % (best_epoch, best_cost))
                        # restore saved variables into tf Variables
                        if output_dir is not None and chkpted and \
                                opt_params['early_stop_mode'] > 0:
                            # save_file exists only if chkpted is True
                            self.saver.restore(sess, save_file)
                            # delete files before break to clean up space
                            shutil.rmtree(os.path.join(output_dir, 'bstmods'),
                                          ignore_errors=True)
                        break
        return epoch
        # END TNDN._train_adam

    # todo: copy generate prediction here and modify it in such a way that's compatible with TNDNs
    # or maybe just edit that... yeah just edit the generate_prediction function so that it checks if it is NDN or TNDN
    # in case of TNDN it edits the self.batch_size to be equal to data_indxs so that it can generate what we want.

    def generate_prediction(self, input_data, data_indxs=None,
                            use_gpu=True, single_batch=False,
                            ffnet_n=-1, layer=-1):
        """Get cost for each output neuron without regularization terms

        Args:
            input_data (time x input_dim numpy array): input to model
            data_indxs (numpy array, optional): indexes of data to use in
                calculating forward pass; if not supplied, all data is used
            use_gpu (True or False): Obvious
            single_batch (True or False): if True, will update the batch_size
                to be equal to len(data_indxs)
            ffnet_n (int, optional): index into `network_list` that specifies
                which FFNetwork to generate the prediction from
            layer (int, optional): index into layers of network_list[ffnet_n]
                that specifies which layer to generate prediction from

        Returns:
            numpy array: pred values from network_list[ffnet_n].layers[layer]

        Raises:
            ValueError: If `layer` index is larger than number of layers in
                network_list[ffnet_n]

        """

        # check input
        if type(input_data) is not list:
            input_data = [input_data]
        self.num_examples = input_data[0].shape[0]
        for temp_data in input_data:
            if temp_data.shape[0] != self.num_examples:
                raise ValueError(
                    'Input data dims must match across input_data.')

        if layer >= len(self.networks[ffnet_n].layers):
            ValueError('This layer does not exist.')
        if data_indxs is None:
            data_indxs = np.arange(self.num_examples)

        # keep original batch size just in case
        original_batch_sz = deepcopy(self.batch_size)

        # change batch_size to be compatible
        # with computations in CaTentLayer graph
        if single_batch:
            new_batch_sz = data_indxs.shape[0]
            self._set_batch_size(new_batch_sz)

        # Generate fake_output data and take care of data-filtering, in case
        # necessary
        self.filter_data = False
        num_outputs = len(self.ffnet_out)
        output_data = [None] * num_outputs
        for nn in range(num_outputs):
            if type(self.networks[ffnet_n].layers[layer]) is CaTentLayer:
                temp_num_outputs = self.networks[ffnet_n].layers[layer].output_dims[1]
            else:
                temp_num_outputs = self.networks[ffnet_n].layers[layer].weights.shape[1]
            output_data[nn] = np.zeros([self.num_examples, temp_num_outputs], dtype='float32')

        # build datasets if using 'iterator' pipeline
        if self.data_pipe_type == 'iterator':
            dataset = self._build_dataset(
                input_data=input_data,
                output_data=output_data,
                indxs=data_indxs,
                training_dataset=False,
                batch_size=self.num_examples)
            # store info on dataset for buiding data pipeline
            self.dataset_types = dataset.output_types
            self.dataset_shapes = dataset.output_shapes
            # build iterator object to access elements from dataset
            iterator = dataset.make_one_shot_iterator()

        # Place graph operations on CPU
        if not use_gpu:
            temp_config = tf.ConfigProto(device_count={'GPU': 0})
            with tf.device('/cpu:0'):
                self._build_graph()
        else:
            temp_config = tf.ConfigProto(device_count={'GPU': 1})
            self._build_graph()

        if single_batch:
            with tf.Session(graph=self.graph, config=temp_config) as sess:

                self._restore_params(sess, input_data, output_data)

                if self.data_pipe_type == 'data_as_var':
                    feed_dict = {self.indices: data_indxs}
                elif self.data_pipe_type == 'feed_dict':
                    feed_dict = self._get_feed_dict(
                        input_data=input_data,
                        batch_indxs=data_indxs)
                elif self.data_pipe_type == 'iterator':
                    # get string handle of iterator
                    iter_handle = sess.run(iterator.string_handle())
                    feed_dict = {self.iterator_handle: iter_handle}

                pred = sess.run(self.networks[ffnet_n].layers[layer].outputs,
                                feed_dict=feed_dict)
        else:
            data_indxs_sz = len(data_indxs)
            pred = np.zeros((data_indxs_sz, temp_num_outputs), dtype='float32')

            # this is the effective batch size with useful info in pred
            _b_tau = original_batch_sz - self.time_spread - self.time_spread//2

            with tf.Session(graph=self.graph, config=temp_config) as sess:
                self._restore_params(sess, input_data, output_data)
                for nn in range(data_indxs_sz//_b_tau + 1):

                    big_end = min(nn*_b_tau + original_batch_sz, data_indxs_sz)
                    small_end = min(self.time_spread + (nn + 1)*_b_tau, data_indxs_sz)

                    big_intvl = np.arange(nn*_b_tau, big_end)
                    small_intvl = np.arange(self.time_spread + nn*_b_tau, small_end)

                    if len(big_intvl) != original_batch_sz:
                        continue

                    if self.data_pipe_type == 'data_as_var':
                        feed_dict = {self.indices: data_indxs[big_intvl]}
                    elif self.data_pipe_type == 'feed_dict':
                        feed_dict = self._get_feed_dict(
                            input_data=input_data,
                            batch_indxs=batch_indxs)
                    elif self.data_pipe_type == 'iterator':
                        # get string handle of iterator
                        iter_handle = sess.run(iterator.string_handle())
                        feed_dict = {self.iterator_handle: iter_handle}

                    pred_tmp = sess.run(
                        self.networks[ffnet_n].layers[layer].outputs, feed_dict=feed_dict)
                    pred[small_intvl, :] = pred_tmp[self.time_spread : -self.time_spread // 2 + 1, :]

            # now do the last remaining part
            self._set_batch_size(len(big_intvl))
            if not use_gpu:
                temp_config = tf.ConfigProto(device_count={'GPU': 0})
                with tf.device('/cpu:0'):
                    self._build_graph()
            else:
                temp_config = tf.ConfigProto(device_count={'GPU': 1})
                self._build_graph()

            with tf.Session(graph=self.graph, config=temp_config) as sess:
                self._restore_params(sess, input_data, output_data)

                if self.data_pipe_type == 'data_as_var':
                    feed_dict = {self.indices: data_indxs[big_intvl]}
                elif self.data_pipe_type == 'feed_dict':
                    feed_dict = self._get_feed_dict(
                        input_data=input_data,
                        batch_indxs=batch_indxs)
                elif self.data_pipe_type == 'iterator':
                    # get string handle of iterator
                    iter_handle = sess.run(iterator.string_handle())
                    feed_dict = {self.iterator_handle: iter_handle}

                pred_tmp = sess.run(
                    self.networks[ffnet_n].layers[layer].outputs, feed_dict=feed_dict)
                pred[small_intvl, :] = pred_tmp[self.time_spread:
                                                self.time_spread + len(small_intvl) + 1, :]

            print('WARNING: discard the first tau time points when single_batch=False... (tau = self.time_spread)')

        # change the batch_size back to its original value
        self._set_batch_size(original_batch_sz)

        return pred
    # END TNDN.generate_prediction

    def copy_model(self, tf_seed=0):
        """Makes an exact copy of model without further elaboration."""

        # Assemble network_list
        target = TNDN(self.network_list, ffnet_out=self.ffnet_out, time_spread=self.time_spread,
                     noise_dist=self.noise_dist, batch_size=self.batch_size, tf_seed=tf_seed)

        target.poisson_unit_norm = self.poisson_unit_norm
        target.data_pipe_type = self.data_pipe_type

        # Copy all the parameters
        for nn in range(self.num_networks):
            for ll in range(self.networks[nn].num_layers):
                target.networks[nn].layers[ll].weights = self.networks[nn].layers[ll].weights.copy()
                target.networks[nn].layers[ll].biases = self.networks[nn].layers[ll].biases.copy()
                target.networks[nn].layers[ll].reg = self.networks[nn].layers[ll].reg.reg_copy()

        return target
    # END TNDN.copy_model()
