"""Temporal Neural deep network"""

from __future__ import print_function
from __future__ import division
from copy import deepcopy

import os
import numpy as np
import tensorflow as tf

from .NDN import NDN
from FFnetwork.ffnetwork import FFNetwork
#from .network import Network
from NDNutils import concatenate_input_dims
from FFnetwork.layer import Layer
from FFnetwork.layer import ConvLayer
from FFnetwork.layer import SepLayer
from FFnetwork.layer import ConvSepLayer
from FFnetwork.layer import AddLayer
from FFnetwork.layer import SpikeHistoryLayer
from FFnetwork.layer import BiConvLayer
from FFnetwork.regularization import Regularization


class tNDN(NDN):

    def __init__(
            self,
            network_list=None,
            noise_dist='poisson',
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

        if network_list is None:
            raise TypeError('Must specify network list.')

        if batch_size is None:
            raise TypeError('Must specify batch size.')

        if time_spread is None:
            raise TypeError('Must specify batch size.')

        # Call __init__() method of super class
        super(tNDN, self).__init__(
            network_list=network_list,
            noise_dist=noise_dist,
            ffnet_out=ffnet_out,
            input_dim_list=input_dim_list,
            tf_seed=tf_seed)

        self.batch_size = batch_size
        self.time_spread = time_spread
    # END tNDN.__init

    def _define_network(self):
        # This code clipped from NDN, where tFFnetworks has to be added

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
                    side_network(
                        scope='side_network_%i' % nn,
                        input_network_params=network_input_params,
                        params_dict=self.network_list[nn]))
            elif self.network_list[nn]['network_type'] == 'temporalFF':
                self.networks.append(
                    tFFNetwork(
                        scope='temporal network_%i' % nn,
                        params_dict=self.network_list[nn]),
                        batch_size=self.batch_size,
                        time_spread=self.time_spread)
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

    # END tNDN._define_network

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

            NT = tf.cast(tf.shape(pred)[0], tf.float32)
            # define cost function
            if self.noise_dist == 'gaussian':
                with tf.name_scope('gaussian_loss'):
                    cost.append(
                        tf.nn.l2_loss(data_out - pred) / NT)
                    unit_cost.append(tf.reduce_mean(tf.square(data_out-pred), axis=0))

            elif self.noise_dist == 'poisson':
                with tf.name_scope('poisson_loss'):

                    if self.poisson_unit_norm is not None:
                        # normalize based on rate * time (number of spikes)
                        cost_norm = tf.multiply(self.poisson_unit_norm, NT)
                    else:
                        cost_norm = NT

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
        # overwrite unit_cost_norm with opt_params value
        self.poisson_unit_norm = opt_params['poisson_unit_norm']

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
    # END tNDN.train

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
        self.batch_size = opt_params['batch_size']
        if self.data_pipe_type != 'data_as_var':
            assert self.batch_size is not None, 'Need to assign batch_size to train.'

        if opt_params['early_stop'] > 0:
            prev_costs = np.multiply(np.ones(opt_params['early_stop']), float('NaN'))

        num_batches_tr = train_indxs.shape[0] // opt_params['batch_size']

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

            # shuffle data before each pass
            train_indxs_perm = np.random.permutation(train_indxs)

            # pass through dataset once
            for batch in range(num_batches_tr):
                if (self.data_pipe_type == 'data_as_var') or (
                        self.data_pipe_type == 'feed_dict'):
                    # get training indices for this batch
                    batch_indxs = train_indxs_perm[
                        batch * opt_params['batch_size']:
                        (batch + 1) * opt_params['batch_size']]

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
                                     batch_tr * opt_params['batch_size']:
                                     (batch_tr + 1) * opt_params['batch_size']]
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
                print('Epoch %04d:  avg train cost = %10.4f,  '
                      'avg test cost = %10.4f,  '
                      'reg penalty = %10.4f'
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
        # END tNDN._train_adam


class tFFNetwork(FFnetwork):
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
        """Constructor for tFFNetwork class"""

        super(tFFNetwork, self).__init__(
            scope=scope,
            input_dims=None,
            params_dict=None)

        self.batch_size = batch_size
        self.time_spread = time_spread
    # END tFFNetwork.__init

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

            elif self.layer_types[nn] == 'temporal_layer':
                self.layers.append(temporal_layer(
                    scope='temporal_layer_%i' % nn,
                    input_dims=layer_sizes[nn],
                    output_dims=layer_sizes[nn+1],
                    batch_size=self.batch_size,
                    normalize_weights=network_params['normalize_weights'][nn],
                    weights_initializer=network_params['weights_initializers'][nn],
                    biases_initializer=network_params['biases_initializers'][nn],
                    reg_initializer=network_params['reg_initializers'][nn],
                    num_inh=network_params['num_inh'][nn],
                    pos_constraint=network_params['pos_constraints'][nn],
                    log_activations=network_params['log_activations']))

            else:
                raise TypeError('Layer type %i not defined.' % nn)

    # END tFFNetwork._define_network


class temporal_layer(object):
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
            my_num_inputs=None,  # this is for convsep
            #activation_func='relu',
            batch_size=None,
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
        if scope is None:
            raise TypeError('Must specify batch size')
        if input_dims is None or output_dims is None:
            raise TypeError('Must specify both input and output dimensions')

        super(temporal_layer, self).__init__(
                scope=scope,
                input_dims=input_dims,
                filter_dims=filter_dims,
                output_dims=num_filters,   # Note difference from layer
                activation_func='lin',
                normalize_weights=normalize_weights,
                weights_initializer=weights_initializer,
                biases_initializer=biases_initializer,
                reg_initializer=reg_initializer,
                num_inh=num_inh,
                pos_constraint=pos_constraint,  # note difference from layer (not anymore)
                log_activations=log_activations)


        # Make input, output, and filter sizes explicit
        if isinstance(input_dims, list):
            while len(input_dims) < 3:
                input_dims.append(1)
        else:
            input_dims = [1, input_dims, 1]
        if isinstance(output_dims, list):
            while len(output_dims) < 3:
                output_dims.append(1)
            num_outputs = np.prod(output_dims)
        else:
            num_outputs = output_dims
            output_dims = [1, output_dims, 1]

        self.input_dims = input_dims[:]
        self.output_dims = output_dims[:]
        # default to have N filts for N outputs in base layer class
        self.num_filters = num_outputs
        if filter_dims is None:
            filter_dims = input_dims
        if my_num_inputs is not None:
            num_inputs = my_num_inputs   # this is for convsep
        else:
            num_inputs = filter_dims[0] * filter_dims[1] * filter_dims[2]
        self.filter_dims = filter_dims[:]

        # create excitatory/inhibitory mask
        if num_inh > num_outputs:
            raise ValueError('Too many inhibitory units designated')
        self.ei_mask = [1] * (num_outputs - num_inh) + [-1] * num_inh

        # save positivity constraint on weights
        self.pos_constraint = pos_constraint
        self.normalize_weights = normalize_weights

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
            init_weights_norm = np.linalg.norm(init_weights, axis=0)
            nonzero_indxs = np.where(init_weights_norm > 0)[0]
            init_weights[:, nonzero_indxs] /= init_weights_norm[nonzero_indxs]

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

    def _normalize_weights(self, ws):
        """"Normalize weights as dictated by normalize_variable"""
        if self.normalize_weights > 0:
            wnorms = tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(ws), axis=0)), 1e-8)
            ws_norm = tf.divide(ws, wnorms)
            # another way of doing this: find the norm along axis=0, then find aaa = np.where(ws_norm > 0) and
            # only divide ws with ws_norm in those indices (because the rest of the indices are zero vectors)
        else:
            # ws = tf.identity(self.weights_var)
            ws_norm = ws
        return ws_norm
    # END Layer._normalize_weights

    def build_graph(self, inputs, params_dict=None):

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            # Define computation
            ws = self._normalize_weights(self.weights_var)

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
    # END Layer.build_graph

    def assign_layer_params(self, sess):
        """Read weights/biases in numpy arrays into tf Variables"""
        sess.run(
            [self.weights_var.initializer, self.biases_var.initializer],
            feed_dict={self.weights_ph: self.weights,
                       self.biases_ph: self.biases})
    # END Layer.assign_layer_params

    def write_layer_params(self, sess):
        """Write weights/biases in tf Variables to numpy arrays"""

        self.weights = sess.run(self.weights_var)
        if self.pos_constraint:
            self.weights = np.maximum(self.weights, 0)
        if self.normalize_weights > 0:
            wnorm = np.sqrt(np.sum(np.square(self.weights), axis=0))
            # wnorm[np.where(wnorm == 0)] = 1
            # self.weights = np.divide(self.weights, wnorm)
            self.weights = np.divide(self.weights, np.maximum(wnorm, 1e-8))
        self.biases = sess.run(self.biases_var)

    # END Layer.write_layer_params

    def define_regularization_loss(self):
        """Wrapper function for building regularization portion of graph"""
        with tf.name_scope(self.scope):
            ws = self._normalize_weights(self.weights_var)
            return self.reg.define_reg_loss(ws)

    def set_regularization(self, reg_type, reg_val):
        """Wrapper function for setting regularization"""
        return self.reg.set_reg_val(reg_type, reg_val)

    def assign_reg_vals(self, sess):
        """Wrapper function for assigning regularization values"""
        self.reg.assign_reg_vals(sess)

    def get_reg_pen(self, sess):
        """Wrapper function for returning regularization penalty dict"""
        return self.reg.get_reg_penalty(sess)
