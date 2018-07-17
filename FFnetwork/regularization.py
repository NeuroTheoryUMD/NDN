"""Module for managing regularization penalties"""

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import FFnetwork.create_reg_matrices as makeRmats


class Regularization(object):
    """Class for handling layer-wise regularization
    
    Attributes:
        vals (dict): values for different types of regularization stored as
            floats
        vals_ph (dict): placeholders for different types of regularization to
            simplify the tf Graph when experimenting with different reg vals
        vals_var (dict): values for different types of regularization stored as
            (un-trainable) tf.Variables
        mats (dict): matrices for different types of regularization stored as
            tf constants
        penalties (dict): tf ops for evaluating different regularization 
            penalties
        input_dims (list): dimensions of layer input size; for constructing reg 
            matrices
        num_outputs (int): dimension of layer output size; for generating 
            target weights in norm2

    """

    _allowed_reg_types = ['l1', 'l2', 'norm2', 'norm2_space', 'norm2_filt',
                          'd2t', 'd2x', 'd2xt', 'local', 'glocal', 'center',
                          'max', 'max_filt', 'max_space']

    def __init__(self, input_dims=None, num_outputs=None, vals=None):
        """Constructor for Regularization class
        
        Args:
            input_dims (int): dimension of input size (for building reg mats)
            num_outputs (int): number of outputs (for normalization in norm2)
            vals (dict, optional): key-value pairs specifying value for each
                type of regularization 

        Raises:
            TypeError: If `input_dims` is not specified
            TypeError: If `num_outputs` is not specified
            
        """

        from copy import deepcopy

        # check input
        if input_dims is None:
            raise TypeError('Must specify `input_dims`')
        if num_outputs is None:
            raise TypeError('Must specify `num_outputs`')

        if isinstance(input_dims, list) is False:
            input_dims = [1, input_dims, 1]

        self.input_dims = input_dims[:]
        self.num_outputs = num_outputs

        # set all default values to None
        none_default = {}
        for reg_type in self._allowed_reg_types:
            none_default[reg_type] = None
        self.vals = deepcopy(none_default)
        self.vals_ph = deepcopy(none_default)
        self.vals_var = deepcopy(none_default)
        self.mats = deepcopy(none_default)
        self.penalties = deepcopy(none_default)

        # read user input
        if vals is not None:
            for reg_type, reg_val in vals.iteritems():
                self.set_reg_val(reg_type, reg_val)
    # END Regularization.__init__

    def set_reg_val(self, reg_type, reg_val):
        """Set regularization value in self.vals dict (doesn't affect a tf 
        Graph until a session is run and `assign_reg_vals` is called)
        
        Args:
            reg_type (str): see `_allowed_reg_types` for options
            reg_val (float): value of regularization parameter
            
        Returns:
            bool: True if `reg_type` has not been previously set
            
        Raises:
            ValueError: If `reg_type` is not a valid regularization type
            ValueError: If `reg_val` is less than 0.0
            
        """

        # check inputs
        if reg_type not in self._allowed_reg_types:
            raise ValueError('Invalid regularization type ''%s''' % reg_type)
        if reg_val < 0.0:
            raise ValueError('`reg_val` must be greater than or equal to zero')

        # determine if this is a new type of regularization
        if self.vals[reg_type] is None:
            new_reg_type = True
        else:
            new_reg_type = False

        self.vals[reg_type] = reg_val

        return new_reg_type
    # END Regularization.set_reg_val

    def assign_reg_vals(self, sess):
        """Update regularization values in default tf Graph"""
        # loop through all types of regularization
        for reg_type, reg_val in self.vals.iteritems():
            # only assign if applicable
            if reg_val is not None:
                sess.run(
                    self.vals_var[reg_type].initializer,
                    feed_dict={self.vals_ph[reg_type]: self.vals[reg_type]})
    # END Regularization.assign_reg_vals

    def define_reg_loss(self, weights):
        """Define regularization loss in default tf Graph"""

        reg_loss = []
        # loop through all types of regularization
        for reg_type, reg_val in self.vals.iteritems():
            # set up reg val variable if it doesn't already exist
            if reg_val is not None:
                with tf.name_scope(reg_type + '_loss'):
                    # use placeholder to initialize Variable for easy
                    # reassignment of reg vals
                    self.vals_ph[reg_type] = tf.placeholder(
                        shape=(),
                        dtype=tf.float32,
                        name=reg_type + '_ph')
                    self.vals_var[reg_type] = tf.Variable(
                        self.vals_ph[reg_type],  # initializer for variable
                        trainable=False,  # no GraphKeys.TRAINABLE_VARS
                        collections=[],   # no GraphKeys.GLOBAL_VARS
                        dtype=tf.float32,
                        name=reg_type + '_param')
                    self.mats[reg_type] = self._build_reg_mats(reg_type)
                    self.penalties[reg_type] = \
                        self._calc_reg_penalty(reg_type, weights)
                reg_loss.append(self.penalties[reg_type])

        # if no regularization, define regularization loss to be zero
        if len(reg_loss) == 0:
            reg_loss.append(tf.constant(0.0, tf.float32, name='zero'))

        return tf.add_n(reg_loss)
    # END Regularization.define_reg_loss

    def _build_reg_mats(self, reg_type):
        """Build regularization matrices in default tf Graph

        Args:
            reg_type (str): see `_allowed_reg_types` for options
        """

        if (reg_type == 'd2t') or (reg_type == 'd2x') or (reg_type == 'd2xt'):
            reg_mat = makeRmats.create_tikhonov_matrix(
                self.input_dims, reg_type)
            name = reg_type + '_laplacian'
        elif (reg_type == 'max') or (reg_type == 'max_filt') or (reg_type == 'max_space'):
            reg_mat = makeRmats.create_maxpenalty_matrix(
                self.input_dims, reg_type)
            name = reg_type + '_reg'
        elif reg_type == 'center':
            reg_mat = makeRmats.create_maxpenalty_matrix(
                self.input_dims, reg_type)
            name = reg_type + '_reg'
        elif reg_type == 'local':
            reg_mat = makeRmats.create_localpenalty_matrix(
                self.input_dims, separable=False)
            name = reg_type + '_reg'
        elif reg_type == 'glocal':
            reg_mat = makeRmats.create_localpenalty_matrix(
                self.input_dims, separable=False, spatial_global=True)
            name = reg_type + '_reg'
        else:
            reg_mat = 0.0
            name = 'lp_placeholder'

        return tf.constant(reg_mat, dtype=tf.float32, name=name)
    # END Regularization._build_reg_mats

    def _calc_reg_penalty(self, reg_type, weights):
        """Calculate regularization penalty for various reg types in default tf 
        Graph"""

        if reg_type == 'l1':
            reg_pen = tf.multiply(
                self.vals_var['l1'],
                tf.reduce_sum(tf.abs(weights)))
        elif reg_type == 'l2':
            reg_pen = tf.multiply(
                self.vals_var['l2'],
                tf.nn.l2_loss(weights))
        elif reg_type == 'norm2':
            reg_pen = tf.multiply(
                self.vals_var['norm2'],
                tf.square(tf.reduce_sum(tf.square(weights))-self.num_outputs))
        elif reg_type == 'max':
            w2 = tf.square(weights)
            reg_pen = tf.multiply(
                self.vals_var['max'],
                tf.trace(tf.matmul(w2,
                                   tf.matmul(self.mats['max'], w2),
                                   transpose_a=True)))
        elif reg_type == 'max_space':
            w2 = tf.square(weights)
            reg_pen = tf.multiply(
                self.vals_var['max_space'],
                tf.trace(tf.matmul(w2,
                                   tf.matmul(self.mats['max_space'], w2),
                                   transpose_a=True)))
        elif reg_type == 'max_filt':
            w2 = tf.square(weights)
            reg_pen = tf.multiply(
                self.vals_var['max_filt'],
                tf.trace(tf.matmul(w2,
                                   tf.matmul(self.mats['max_filt'], w2),
                                   transpose_a=True)))
        elif reg_type == 'd2t':
            reg_pen = tf.multiply(
                self.vals_var['d2t'],
                tf.reduce_sum(tf.square(
                    tf.matmul(self.mats['d2t'], weights))))
        elif reg_type == 'd2x':
            reg_pen = tf.multiply(
                self.vals_var['d2x'],
                tf.reduce_sum(tf.square(
                    tf.matmul(self.mats['d2x'], weights))))
        elif reg_type == 'd2xt':
            reg_pen = tf.multiply(
                self.vals_var['d2xt'],
                tf.reduce_sum(tf.square(
                    tf.matmul(self.mats['d2xt'], weights))))
        elif reg_type == 'local':
            w2 = tf.square(weights)
            reg_pen = tf.multiply(
                self.vals_var['local'],
                tf.trace(tf.matmul(w2,
                                   tf.matmul(self.mats['local'], w2),
                                   transpose_a=True)))
        elif reg_type is 'glocal':
            w2 = tf.square(weights)
            reg_pen = tf.multiply(
                self.vals_var['glocal'],
                tf.trace(tf.matmul(w2, tf.matmul(self.mats['glocal'], w2),
                                   transpose_a=True)))
        elif reg_type == 'center':
            reg_pen = tf.multiply(
                self.vals_var['center'],
                tf.trace(tf.matmul(weights,
                                   tf.matmul(self.mats['center'], weights),
                                   transpose_a=True)))

        else:
            reg_pen = tf.constant(0.0)
        return reg_pen
    # END Regularization._calc_reg_penalty

    def get_reg_penalty(self, sess):
        """Build dictionary that contains regularization penalty from each 
        regularization type"""

        reg_dict = {}
        for reg_type, reg_val in self.vals.iteritems():
            if reg_val is not None:
                reg_pen = sess.run(self.penalties[reg_type])
            else:
                reg_pen = 0.0
            reg_dict[reg_type] = reg_pen

        return reg_dict
    # END Regularization.get_reg_penalty

    def reg_copy(self):
        """Copy regularization to new structure"""
        reg_target = Regularization(
            input_dims=self.input_dims,
            num_outputs=self.num_outputs)
        reg_target.vals = self.vals.copy()
        reg_target.mats = {}

        return reg_target
    # END Regularization.reg_copy
# END Regularization


class SepRegularization(Regularization):
    """Child class that adjusts regularization for separable layers"""

    def __init__(self, input_dims=None, num_outputs=None, vals=None):
        """Constructor for Sep_Regularization object
        
        Args:
            input_dims (int): dimension of input size (for building reg mats)
            num_outputs (int): number of outputs (for normalization in norm2)
            vals (dict, optional): key-value pairs specifying value for each
                type of regularization 
                
        """

        super(SepRegularization, self).__init__(
            input_dims=input_dims,
            num_outputs=num_outputs,
            vals=vals)
    # END SepRegularization.__init__

    def _build_reg_mats(self, reg_type):
        """Build regularization matrices in default tf Graph

        Args:
            reg_type (str): see `_allowed_reg_types` for options
            
        """

        if reg_type == 'd2t':
            reg_mat = makeRmats.create_tikhonov_matrix(
                [self.input_dims[0], 1, 1], reg_type)
            name = reg_type + '_laplacian'
        elif reg_type == 'd2x':
            reg_mat = makeRmats.create_tikhonov_matrix(
                [1, self.input_dims[1], self.input_dims[2]], reg_type)
            name = reg_type + '_laplacian'
        elif reg_type == 'd2xt':
            raise TypeError('d2xt does not work with a separable layer.')
        elif reg_type == 'max':
            raise ValueError('Cannot use max regularization with a separable layer.')
        elif reg_type == 'max_filt':
            reg_mat = makeRmats.create_maxpenalty_matrix(
                [self.input_dims[0], 1, 1], 'max')
            name = reg_type + '_reg'
        elif reg_type == 'max_space':
            reg_mat = makeRmats.create_maxpenalty_matrix(
                [self.input_dims[1]*self.input_dims[2], 1, 1], 'max')
            name = reg_type + '_reg'
        elif reg_type == 'center':
            reg_mat = makeRmats.create_maxpenalty_matrix(
                [1, self.input_dims[1], self.input_dims[2]], 'center')
            name = reg_type + '_reg'
        elif reg_type == 'local':
            reg_mat = makeRmats.create_localpenalty_matrix(
                self.input_dims, separable=True)
            name = reg_type + '_reg'
        elif reg_type == 'glocal':
            raise TypeError('glocal regularization not supported with a separable layer.')
        else:
            reg_mat = 0.0
            name = 'lp_placeholder'

        return tf.constant(reg_mat, dtype=tf.float32, name=name)
    # END SepRegularization._build_reg_mats

    def _calc_reg_penalty(self, reg_type, weights):
        """Calculate regularization penalty for various reg types in default tf
        Graph"""

        if reg_type == 'l1':
            reg_pen = tf.multiply(
                self.vals_var['l1'],
                tf.reduce_sum(tf.abs(weights)))
        elif reg_type == 'l2':
            reg_pen = tf.multiply(
                self.vals_var['l2'],
                tf.nn.l2_loss(weights))
        elif reg_type == 'norm2':
            reg_pen = tf.multiply(
                self.vals_var['norm2'],
                tf.square(tf.reduce_sum(tf.square(weights))-self.num_outputs))
        elif reg_type == 'norm2_filt':
            wfilt = tf.slice(weights, [0, 0], [self.input_dims[0],
                                               self.num_outputs])
            reg_pen = tf.multiply(
                self.vals_var['norm2_filt'],
                tf.square(tf.reduce_sum(tf.square(wfilt))-self.num_outputs))
        elif reg_type == 'norm2_space':
            wspace = tf.slice(weights, [self.input_dims[0], 0],
                              [self.input_dims[1]*self.input_dims[2],
                               self.num_outputs])
            reg_pen = tf.multiply(
                self.vals_var['norm2_space'],
                tf.square(tf.reduce_sum(tf.square(wspace))-self.num_outputs))

        elif reg_type == 'max_space':
            wspace2 = tf.square(tf.slice(weights, [self.input_dims[0], 0],
                          [self.input_dims[1]*self.input_dims[2],
                           self.num_outputs]))
            reg_pen = tf.multiply(
                self.vals_var['max_space'], tf.trace(tf.matmul(
                    wspace2, tf.matmul(self.mats['max_space'], wspace2),
                    transpose_a=True)))

        elif reg_type == 'max_filt':
            wfilt2 = tf.square(tf.slice(
                weights, [0, 0], [self.input_dims[0], self.num_outputs]))
            reg_pen = tf.multiply(
                self.vals_var['max_filt'],
                tf.trace(tf.matmul(wfilt2,
                                   tf.matmul(self.mats['max_filt'],
                                             wfilt2), transpose_a=True)))

        elif reg_type == 'd2t':
            wt = tf.slice(weights, [0, 0], [self.input_dims[0],
                                            self.num_outputs])
            reg_pen = tf.multiply(self.vals_var['d2t'],
                                  tf.reduce_sum(tf.square(tf.matmul(self.mats['d2t'], wt))))

        elif reg_type == 'd2x':
            wspace = tf.slice(weights, [self.input_dims[0], 0],
                              [self.input_dims[1]*self.input_dims[2],
                               self.num_outputs])
            reg_pen = tf.multiply(self.vals_var['d2x'],
                                  tf.reduce_sum(tf.square(
                                      tf.matmul(self.mats['d2x'], wspace))))
        elif reg_type == 'center':
            wspace = tf.slice(weights, [self.input_dims[0], 0],
                              [self.input_dims[1]*self.input_dims[2],
                               self.num_outputs])
            reg_pen = tf.multiply(
                self.vals_var['center'],
                tf.trace(tf.matmul(wspace,
                                   tf.matmul(self.mats['center'], wspace),
                                   transpose_a=True)))
        elif reg_type == 'local':
            wspace2 = tf.square(tf.slice(weights, [self.input_dims[0], 0],
                          [self.input_dims[1]*self.input_dims[2],
                           self.num_outputs]))
            reg_pen = tf.multiply(
                self.vals_var['local'],
                tf.trace(tf.matmul(wspace2, tf.matmul(self.mats['local'], wspace2),
                                   transpose_a=True)))
        elif reg_type == 'd2xt':
            raise TypeError('d2xt does not work with a separable layer.')
        else:
            reg_pen = tf.constant(0.0)
        return reg_pen
    # END Sep_Regularization._calc_reg_penalty

    def reg_copy(self):
        """Copy regularization to new structure"""

        reg_target = SepRegularization(
            input_dims=self.input_dims,
            num_outputs=self.num_outputs)
        reg_target.vals = self.vals.copy()
        #reg_target.mats = self.mats.copy()
        reg_target.mats = {}

        return reg_target
    # END SepRegularization.reg_copy
