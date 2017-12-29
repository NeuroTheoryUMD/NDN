"""Basic network-building tools"""

from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np
import tensorflow as tf


class Network(object):
    """Base class for neural networks"""

    _allowed_learning_algs = ['adam', 'lbfgs']
    _log_min = 1e-5  # constant to add to all arguments to logarithms

    def __init__(self):
        """Constructor for Network class; model architecture should be defined
        elsewhere
        """

        self.num_examples = 0

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

    def _define_loss(self):
        """Loss function that will be used to optimize model parameters"""
        raise NotImplementedError

    def _define_optimizer(self, learning_alg='adam', learning_rate=1e-3,
                          var_list=None):
        """Define one step of the optimization routine"""

        if learning_alg == 'adam':
            self.train_step = tf.train.AdamOptimizer(learning_rate). \
                minimize(self.cost_penalized, var_list=var_list)
        elif learning_alg == 'lbfgs':
            self.train_step = tf.contrib.opt.ScipyOptimizerInterface(
                self.cost_penalized,
                var_list=var_list,
                method='L-BFGS-B',
                options={
                    'maxiter': 10000,
                    'disp': False})
    # END _define_optimizer

    def train(
            self,
            input_data=None,
            output_data=None,
            train_indxs=None,
            test_indxs=None,
            fit_parameter_list=None,
            learning_alg='lbfgs',
            learning_rate=1e-3,
            use_gpu=False,
            batch_size=128,
            epochs_training=10000,
            epochs_disp=None,
            epochs_ckpt=None,
            epochs_early_stop=None,
            epochs_summary=None,
            early_stop=False,
            output_dir=None):
        """Network training function

        Args:
            input_data (list): input to network; each element should be a 
                time x input_dim numpy array
            output_data (list): desired output of network; each element should
                be a time x output_dim numpy array
            train_indxs (numpy array, optional): subset of data to use for 
                training
            test_indxs (numpy array, optional): subset of data to use for 
                testing; if available these are used when displaying updates,
                and are also the indices used for early stopping if enabled
            fit_parameter_list (list-of-lists, optional): default none
                Generated by 'fit_variables' (if not none) to reference which
                variables in the model to fit.
            learning_alg (str, optional): algorithm used for learning
                parameters.
                ['lbfgs'] | 'adam'
            learning_rate (float, optional): learning rate used by the
                gradient descent-based optimizers ('adam'). Default is 1e-3.
            use_gpu (bool):
            batch_size (int, optional): batch size used by the gradient
                descent-based optimizers (adam).
            epochs_training (int, optional): number of epochs for gradient 
                descent-based optimizers
            epochs_disp (int, optional): number of epochs between updates to 
                the console
            epochs_ckpt (int, optional): number of epochs between saving 
                checkpoint files
            epochs_early_stop (int, optional): number of epochs between checks
                for early stopping
            epochs_summary (int, optional): number of epochs between saving
                network summary information
            early_stop (bool, optional): if True, training exits when the
                cost function evaluated on test_indxs begins to increase
            output_dir (str, optional): absolute path for saving checkpoint
                files and summary files; must be present if either epochs_ckpt  
                or epochs_summary is not 'None'. If `output_dir` is not 'None',
                the graph will automatically be saved.

        Returns:
            epoch (int): number of total training epochs

        Raises:
            ValueError: If `input_data` and `output_data` don't share time dim
            ValueError: If data time dim doesn't match that specified in model
            ValueError: If `epochs_ckpt` is not None and output_dir is 'None'
            ValueError: If `epochs_summary` is not 'None' and `output_dir` is 
                'None'
            ValueError: If `early_stop` is True and `test_indxs` is 'None'

        """

        # check input
        if type(input_data) is not list:
            input_data = [input_data]
        if type(output_data) is not list:
            output_data = [output_data]
        self.num_examples = input_data[0].shape[0]
        for temp_data in input_data:
            if temp_data.shape[0] != self.num_examples:
                raise ValueError('Input data dims must match across input_data.')
        for temp_data in output_data:
            if temp_data.shape[0] != self.num_examples:
                raise ValueError('Output data dims must match model values')
        if epochs_ckpt is not None and output_dir is None:
            raise ValueError('output_dir must be specified to save model')
        if epochs_summary is not None and output_dir is None:
            raise ValueError('output_dir must be specified to save summaries')
        if early_stop and test_indxs is None:
            raise ValueError('test_indxs must be specified for early stopping')

        if train_indxs is None:
            train_indxs = np.arange(self.num_examples)

        # Build graph: self.build_graph must be defined in child of network
        self._build_graph(learning_alg, learning_rate, use_gpu)

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:

            # handle output directories
            train_writer = None
            test_writer = None
            if output_dir is not None:

                # remake checkpoint directory
                if epochs_ckpt is not None:
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
                    summary_dir_train, sess.graph)

                # remake testing summary directories
                summary_dir_test = os.path.join(
                    output_dir, 'summaries', 'test')
                if test_indxs is not None:
                    if os.path.isdir(summary_dir_test):
                        tf.gfile.DeleteRecursively(summary_dir_test)
                    os.makedirs(summary_dir_test)
                    test_writer = tf.summary.FileWriter(
                        summary_dir_test, sess.graph)

            # Generate fit_parameter_list for fitting if fit_parameter_list
            # (if relevant)
            var_list = self._build_fit_variable_list(fit_parameter_list)

            with tf.variable_scope('optimizer'):
                self._define_optimizer(var_list)

            # overwrite initialized values of network with stored values
            self._restore_params(sess, input_data, output_data)

            # select learning algorithm
            if learning_alg == 'adam':
                epoch = self._train_adam(
                    sess=sess,
                    train_writer=train_writer,
                    test_writer=test_writer,
                    train_indxs=train_indxs,
                    test_indxs=test_indxs,
                    batch_size=batch_size,
                    epochs_training=epochs_training,
                    epochs_disp=epochs_disp,
                    epochs_ckpt=epochs_ckpt,
                    epochs_early_stop=epochs_early_stop,
                    epochs_summary=epochs_summary,
                    early_stop=early_stop,
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
            batch_size=None,
            epochs_training=None,
            epochs_disp=None,
            epochs_ckpt=None,
            epochs_early_stop=None,
            epochs_summary=None,
            early_stop=None,
            output_dir=None):
        """Training function for adam optimizer to clean up code in `train`"""

        num_batches = train_indxs.shape[0] // batch_size

        if early_stop:
            prev_cost = float('Inf')

        # start training loop
        for epoch in range(epochs_training):

            # shuffle data before each pass
            train_indxs_perm = np.random.permutation(train_indxs)

            # pass through dataset once
            for batch in range(num_batches):
                # get training indices for this batch
                batch_indxs = train_indxs_perm[
                              batch * batch_size:
                              (batch + 1) * batch_size]
                # one step of optimization routine
                sess.run(
                    self.train_step,
                    feed_dict={self.indices: batch_indxs})

            # print training updates
            if epochs_disp is not None and \
                    (epoch % epochs_disp == epochs_disp - 1 or epoch == 0):

                cost = sess.run(
                    self.cost,
                    feed_dict={self.indices: train_indxs_perm})
                print('\nEpoch %03d:' % epoch)
                print('   train cost = %2.5f' % cost)

                # print additional testing info
                if test_indxs is not None:
                    cost_test = sess.run(
                        self.cost,
                        feed_dict={self.indices: test_indxs})
                    print('   test cost = %2.5f' % cost_test)

            # save model checkpoints
            if epochs_ckpt is not None and \
                    (epoch % epochs_ckpt == epochs_ckpt - 1 or epoch == 0):
                save_file = os.path.join(
                    output_dir, 'ckpts',
                    str('epoch_%05g.ckpt' % epoch))
                self.checkpoint_model(sess, save_file)

            # save model summaries
            if epochs_summary is not None and \
                    (epoch % epochs_summary == epochs_summary - 1
                     or epoch == 0):
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

            # check for early stopping
            if early_stop and \
                    epoch % epochs_early_stop == epochs_early_stop - 1:

                cost_test = sess.run(
                    self.cost,
                    feed_dict={self.indices: test_indxs})

                if cost_test >= prev_cost:

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
                    prev_cost = cost_test

        return epoch
    # END _train_adam

    def _restore_params(self, sess, input_data, output_data):
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
