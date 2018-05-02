"""Basic network-building tools"""

from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np
import tensorflow as tf

# supress INFO log
tf.logging.set_verbosity(tf.logging.FATAL)


class Network(object):
    """Base class for neural networks"""

    _allowed_learning_algs = ['adam', 'lbfgs']
    _log_min = 1e-6  # constant to add to all arguments to logarithms

    def __init__(self):
        """Constructor for Network class; model architecture should be defined 
        elsewhere"""

        self.num_examples = 0
        self.filter_data = False
    # END Network.__init__

    def _initialize_data_pipeline(self):
        """Define pipeline for feeding data into model"""

        # define indices placeholder to specify subset of data
        self.indices = tf.placeholder(
            dtype=tf.int32,
            shape=None,
            name='indices_ph')

        # INPUT DATA
        self.data_in_ph = [None] * len(self.input_sizes)
        self.data_in_var = [None] * len(self.input_sizes)
        self.data_in_batch = [None] * len(self.input_sizes)
        for i, input_size in enumerate(self.input_sizes):
            # reduce input_sizes to single number if 3-D
            num_inputs = np.prod(input_size)
            # placeholders for data
            self.data_in_ph[i] = tf.placeholder(
                dtype=tf.float32,
                #shape=[self.num_examples, input_size],
                shape=[self.num_examples, num_inputs],
                name='input_ph_%02d' % i)
            # turn placeholders into variables so they get put on GPU
            self.data_in_var[i] = tf.Variable(
                self.data_in_ph[i],  # initializer for Variable
                trainable=False,     # no GraphKeys.TRAINABLE_VARS
                collections=[],      # no GraphKeys.GLOBAL_VARS
                name='input_var_%02d' % i)
            # use selected subset of data
            self.data_in_batch[i] = tf.gather(
                self.data_in_var[i],
                self.indices,
                name='input_batch_%02d' % i)

        # OUTPUT DATA
        self.data_out_ph = [None] * len(self.output_size)
        self.data_out_var = [None] * len(self.output_size)
        self.data_out_batch = [None] * len(self.output_size)
        for i, output_size in enumerate(self.output_size):
            # placeholders for data
            self.data_out_ph[i] = tf.placeholder(
                dtype=tf.float32,
                shape=[self.num_examples, output_size],
                name='output_ph_%02d' % i)
            # turn placeholders into variables so they get put on GPU
            self.data_out_var[i] = tf.Variable(
                self.data_out_ph[i],  # initializer for Variable
                trainable=False,      # no GraphKeys.TRAINABLE_VARS
                collections=[],       # no GraphKeys.GLOBAL_VARS
                name='output_var_%02d' % i)
            # use selected subset of data
            self.data_out_batch[i] = tf.gather(
                self.data_out_var[i],
                self.indices,
                name='output_batch_%02d' % i)

        # DATA FILTERS
        if self.filter_data:
            self.data_filter_ph = [None] * len(self.output_size)
            self.data_filter_var = [None] * len(self.output_size)
            self.data_filter_batch = [None] * len(self.output_size)
            for ii, output_size in enumerate(self.output_size):
                # placeholders for data
                self.data_filter_ph[i] = tf.placeholder(
                    dtype=tf.float32,
                    shape=[self.num_examples, output_size],
                    name='data_filter_ph_%02d' % i)
                # turn placeholders into variables so they get put on GPU
                self.data_filter_var[i] = tf.Variable(
                    self.data_filter_ph[i],  # initializer for Variable
                    trainable=False,  # no GraphKeys.TRAINABLE_VARS
                    collections=[],  # no GraphKeys.GLOBAL_VARS
                    name='output_filter_%02d' % i)
                # use selected subset of data
                self.data_filter_batch[i] = tf.gather(
                    self.data_filter_var[i],
                    self.indices,
                    name='output_filter_%02d' % i)
    # END Network._initialize_data_pipeline

    def _define_loss(self):
        """Loss function that will be used to optimize model parameters"""
        raise NotImplementedError

    def _define_optimizer(self, learning_alg='adam', opt_params=None,
                          var_list=None):
        """Define one step of the optimization routine
        L-BGFS algorithm described 
        https://docs.scipy.org/doc/scipy-0.18.1/reference/optimize.minimize-lbfgsb.html
        """

        if learning_alg == 'adam':
            self.train_step = tf.train.AdamOptimizer(
                learning_rate=opt_params['learning_rate'],
                beta1=opt_params['beta1'],
                beta2=opt_params['beta2'],
                epsilon=opt_params['epsilon']). \
                minimize(self.cost_penalized, var_list=var_list)
        elif learning_alg == 'lbfgs':
            self.train_step = tf.contrib.opt.ScipyOptimizerInterface(
                self.cost_penalized,
                var_list=var_list,
                method='L-BFGS-B',
                options={
                    'maxiter': opt_params['maxiter'],
                    'gtol': opt_params['grad_tol'],
                    'ftol': opt_params['func_tol'],
                    'eps': opt_params['eps'],
                    'disp': opt_params['display']})
    # END _define_optimizer

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

        Returns:
            int: number of total training epochs

        Raises:
            ValueError: If `input_data` and `output_data` don't share time dim
            ValueError: If data time dim doesn't match that specified in model
            ValueError: If `epochs_ckpt` value in `opt_params` is not `None` 
                and `output_dir` is `None`
            ValueError: If `epochs_summary` in `opt_params` is not `None` and 
                `output_dir` is `None`
            ValueError: If `early_stop` >0 and `test_indxs` is 'None'

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
        opt_params = self.optimizer_defaults( opt_params, learning_alg )

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
            if opt_params['early_stop'] > 0 and opt_params['epochs_early_stop'] is None:
                opt_params['epochs_early_stop'] = 1

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

            # select learning algorithm
            if learning_alg == 'adam':
                epoch = self._train_adam(
                    sess=sess,
                    train_writer=train_writer,
                    test_writer=test_writer,
                    train_indxs=train_indxs,
                    test_indxs=test_indxs,
                    opt_params=opt_params,
                    output_dir=output_dir)
            elif learning_alg == 'lbfgs':
                self.train_step.minimize(
                    sess, feed_dict={self.indices: train_indxs})
                epoch = float('NaN')
            else:
                raise ValueError('Invalid learning algorithm')

            # write out weights/biases to numpy arrays before session closes
            self._write_model_params(sess)

        return epoch
    # END train

    def _train_adam(
            self,
            sess=None,
            train_writer=None,
            test_writer=None,
            train_indxs=None,
            test_indxs=None,
            opt_params=None,
            output_dir=None):
        """Training function for adam optimizer to clean up code in `train`"""

        epochs_training = opt_params['epochs_training']
        epochs_ckpt = opt_params['epochs_ckpt']
        epochs_early_stop = opt_params['epochs_early_stop']
        early_stop_pool = opt_params['early_stop']
        epochs_summary = opt_params['epochs_summary']

        num_batches = train_indxs.shape[0] // opt_params['batch_size']

        if opt_params['early_stop'] > 0:
            prev_cost = np.multiply(np.ones([early_stop_pool]), float('Inf'))

        if opt_params['run_diagnostics']:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        # start training loop
        for epoch in range(epochs_training):

            # shuffle data before each pass
            train_indxs_perm = np.random.permutation(train_indxs)

            # pass through dataset once
            for batch in range(num_batches):
                # get training indices for this batch
                batch_indxs = train_indxs_perm[
                              batch * opt_params['batch_size']:
                              (batch + 1) * opt_params['batch_size']]
                # one step of optimization routine
                sess.run(
                    self.train_step,
                    feed_dict={self.indices: batch_indxs})

            # print training updates
            display_output = False
            if opt_params['display'] is not None and \
                    (epoch % opt_params['display'] == opt_params['display'] - 1
                     or epoch == 0):
                display_output = True
                cost = sess.run(
                    self.cost,
                    feed_dict={self.indices: train_indxs_perm})

                if test_indxs is not None:
                    cost_test = sess.run(
                        self.cost,
                        feed_dict={self.indices: test_indxs})
                    reg_pen = sess.run(
                        self.cost_reg,
                        feed_dict={self.indices: train_indxs_perm})

                    # print additional testing info
                    print('Epoch %04d:  train cost = %10.4f, test cost = %10.4f, reg penalty = %10.4f'
                          % (epoch, cost, cost_test, reg_pen))
                else:
                    print('Epoch %04d:  train cost = %2.5f' % (epoch, cost))

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
                if not display_output and opt_params['display'] is not None:
                    print('Epoch %03d:' % epoch)

                # model summary
                print('Writing train summary')
                if opt_params['run_diagnostics']:
                    summary = sess.run(
                        self.merge_summaries,
                        feed_dict={self.indices: train_indxs},
                        options=run_options,
                        run_metadata=run_metadata)
                    train_writer.add_run_metadata(
                      run_metadata, 'epoch_%d' % epoch)
                else:
                    summary = sess.run(
                        self.merge_summaries,
                        feed_dict={self.indices: train_indxs})
                train_writer.add_summary(summary, epoch)
                train_writer.flush()

                if test_indxs is not None:
                    print('Writing test summary')
                    if opt_params['run_diagnostics']:
                        summary = sess.run(
                            self.merge_summaries,
                            feed_dict={self.indices: test_indxs},
                            options=run_options,
                            run_metadata=run_metadata)
                        test_writer.add_run_metadata(
                            run_metadata, 'epoch_%d' % epoch)
                    else:
                        summary = sess.run(
                            self.merge_summaries,
                            feed_dict={self.indices: test_indxs})
                test_writer.add_summary(summary, epoch)
                test_writer.flush()

            # check for early stopping
            if opt_params['early_stop'] > 0 and \
                    epoch % epochs_early_stop == epochs_early_stop - 1:

                cost_test = sess.run(
                    self.cost,
                    feed_dict={self.indices: test_indxs})

                if cost_test >= np.max(prev_cost):

                    # save model checkpoint if desired and necessary
                    if epochs_ckpt is not None and \
                            epochs_ckpt != epochs_early_stop:
                        save_file = os.path.join(
                            output_dir, 'ckpts',
                            str('epoch_%05g.ckpt' % epoch))
                        self.checkpoint_model(sess, save_file)

                    # save model summaries if desired and necessary
                    if epochs_summary is not None and \
                            epochs_summary != epochs_early_stop:
                        summary = sess.run(
                            self.merge_summaries,
                            feed_dict={self.indices: train_indxs})
                        train_writer.add_summary(summary, epoch)
                        print('Writing train summary')
                        if test_indxs is not None:
                            summary = sess.run(
                                self.merge_summaries,
                                feed_dict={self.indices: test_indxs})
                            test_writer.add_summary(summary, epoch)
                            print('Writing test summary')

                    break  # out of epochs loop
                else:
                    prev_cost[(epoch // epochs_early_stop) % early_stop_pool] = cost_test

        return epoch
    # END _train_adam

    def _restore_params(self, sess, input_data, output_data,
                        data_filters=None):
        """Restore model parameters from numpy matrices and update
        regularization values from list. This function is called by any other 
        function that needs to initialize a new session to run parts of the 
        graph."""

        # initialize all parameters randomly
        sess.run(self.init)

        # check input
        if type(input_data) is not list:
            input_data = [input_data]
        if type(output_data) is not list:
            output_data = [output_data]

        # initialize input/output data
        for i, temp_data in enumerate(input_data):
            sess.run(self.data_in_var[i].initializer,
                     feed_dict={self.data_in_ph[i]: temp_data})
        for i, temp_data in enumerate(output_data):
            sess.run(self.data_out_var[i].initializer,
                     feed_dict={self.data_out_ph[i]: temp_data})
            if self.filter_data:
                sess.run(self.data_filter_var[i].initializer,
                         feed_dict={self.data_filter_ph[i]: data_filters[i]})

        # overwrite randomly initialized values of model with stored values
        self._assign_model_params(sess)

        # update regularization parameter values
        self._assign_reg_vals(sess)

    def _assign_model_params(self, sess):
        """Assigns parameter values previously stored in numpy arrays to 
        tf Variables in model; function needs to be implemented by specific 
        model"""
        raise NotImplementedError()

    def _assign_reg_vals(self, sess):
        """Loops through all current regularization penalties and updates  
        parameter values in the tf Graph; needs to be implemented by specific 
        model"""
        raise NotImplementedError()

    def checkpoint_model(self, sess, save_file):
        """Checkpoint model parameters in tf Variables

        Args:
            sess (tf.Session object): current session object to run graph
            save_file (str): full path to output file

        """

        if not os.path.isdir(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))

        self.saver.save(sess, save_file)
        print('Model checkpointed to %s' % save_file)

    def restore_model(self, save_file, input_data, output_data):
        """Restore previously checkpointed model parameters in tf Variables 

        Args:
            sess (tf.Session object): current session object to run graph
            save_file (str): full path to saved model
            input_data (time x input_dim numpy array): input to network
            output_data (time x output_dim numpy array): desired output of 
                network

        Raises:
            ValueError: If `save_file` is not a valid filename

        """

        if not os.path.isfile(save_file + '.meta'):
            raise ValueError(str('%s is not a valid filename' % save_file))

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:

            # initialize tf params in new session
            self._restore_params(sess, input_data, output_data)
            # restore saved variables into tf Variables
            self.saver.restore(sess, save_file)
            # write out weights/biases to numpy arrays before session closes
            self.network.write_model_params(sess)

    def save_model(self, save_file):
        """Save full network object using dill (extension of pickle)

        Args:
            save_file (str): full path to output file

        """

        import dill

        sys.setrecursionlimit(10000)  # for dill calls to pickle

        if not os.path.isdir(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))

        with file(save_file, 'wb') as f:
            dill.dump(self, f)
        print('Model pickled to %s' % save_file)

    @classmethod
    def load_model(cls, save_file):
        """Restore previously saved network object 

        Args:
            save_file (str): full path to saved model

        Raises:
            ValueError: If `save_file` is not a valid filename

        """

        import dill

        if not os.path.isfile(save_file):
            raise ValueError(str('%s is not a valid filename' % save_file))

        with file(save_file, 'rb') as f:
            return dill.load(f)

    @classmethod
    def optimizer_defaults(cls, opt_params, learning_alg):
        """Sets defaults for different optimizers
        
        In the `opt_params` dictionary, the `display` and `use_gpu` keys are 
        available for all optimizers. The following keys are used exclusively
        for lbfgs: `max_iter`, `func_tol`, `grad_tol` and `eps`. The remaining 
        keys are all specific to the adam optimizer.
        
        Args:
            opt_params: dictionary with optimizer-specific parameters
            opt_params['display'] (int, optional): For adam, this defines the 
                number of epochs between updates to the console. Becomes 
                boolean for lbfgs, prints detailed optimizer info for each 
                iteration.
                DEFAULT: 0/False
            opt_params['use_gpu'] (bool, optional): `True` to fit model on gpu.
                DEFAULT: False
            opt_params['max_iter'] (int, optional): maximum iterations for  
                lbfgs algorithm.
                DEFAULT: 500
            opt_params['func_tol'] (float, optional): see lbfgs method in SciPy 
                optimizer. 
                DEFAULT: 2.22e-09
            opt_params['grad_tol'] (float, optional): see lbfgs method in SciPy 
                optimizer. 
                DEFAULT: 1e-05
            opt_params['eps'] (float, optional): see lbfgs method in SciPy 
                optimizer. 
                DEFAULT: 1e-08
            opt_params['learning_rate'] (float, optional): learning rate used 
                by adam. 
                DEFAULT: 1e-3.
            opt_params['batch_size'] (int, optional): number of data points to 
                use for each iteration of training.
                DEFAULT: 128
            opt_params['epochs_training'] (int, optional): max number of 
                epochs.
                DEFAULT: 100
            opt_params['epochs_ckpt'] (int, optional): number of epochs between 
                saving checkpoint files.
                DEFAULT: `None`
            opt_params['epochs_early_stop'] (int, optional): number of epochs 
                between checks for early stopping.
                DEFAULT: `None`
            opt_params['early_stop'] (int, optional): if greater than zero, training
                exits when the cost function evaluated on test_indxs is not lower than
                the maximum over that many previous checks
                DEFAULT: 0
            opt_params['beta1'] (float, optional): beta1 (1st momentum term) for Adam
                DEFAULT: 0.9
            opt_params['beta2'] (float, optional): beta2 (2nd momentum term) for Adam
                DEFAULT: 0.999
            opt_params['epsilon'] (float, optional): epsilon parameter in Adam optimizer
                DEFAULT: 1e-4 (note normal Adam default is 1e-8)
            opt_params['epochs_summary'] (int, optional): number of epochs
                between saving network summary information.
                DEFAULT: `None`
            opt_params['run_diagnostics'] (bool, optional): `True` to record
                compute time and memory usage of tensorflow ops during training
                and testing. `epochs_summary` must not be `None`.
                DEFAULT: `False`
            learning_alg (str): 'adam' and 'lbfgs' currently supported
        """

        # Non-optimizer specific defaults
        if 'display' not in opt_params:
            opt_params['display'] = None
        if 'use_gpu' not in opt_params:
            opt_params['use_gpu'] = False

        if learning_alg is 'adam':
            if 'learning_rate' not in opt_params:
                opt_params['learning_rate'] = 1e-3
            if 'batch_size' not in opt_params:
                opt_params['batch_size'] = 128
            if 'epochs_training' not in opt_params:
                opt_params['epochs_training'] = 100
            if 'epochs_ckpt' not in opt_params:
                opt_params['epochs_ckpt'] = None
            if 'epochs_early_stop' not in opt_params:
                opt_params['epochs_early_stop'] = None
            if 'epochs_summary' not in opt_params:
                opt_params['epochs_summary'] = None
            if 'early_stop' not in opt_params:
                opt_params['early_stop'] = 0
            if 'beta1' not in opt_params:
                opt_params['beta1'] = 0.9
            if 'beta2' not in opt_params:
                opt_params['beta2'] = 0.999
            if 'epsilon' not in opt_params:
                opt_params['epsilon'] = 1e-4
            if 'run_diagnostics' not in opt_params:
                opt_params['run_diagnostics'] = False

        else:  # lbfgs
            if 'maxiter' not in opt_params:
                opt_params['maxiter'] = 500
            # The iteration will stop when
            # max{ | proj g_i | i = 1, ..., n} <= pgtol
            # where pg_i is the i-th component of the projected gradient.
            if 'func_tol' not in opt_params:
                opt_params['func_tol'] = 2.220446049250313e-09  #?
            if 'grad_tol' not in opt_params:
                opt_params['grad_tol'] = 1e-05
            if 'eps' not in opt_params:
                opt_params['eps'] = 1e-08
            # Convert display variable to boolean
            if opt_params['display'] is None:
                opt_params['display'] = False
            else:
                opt_params['display'] = True

        return opt_params
    # END network.optimizer_defaults
