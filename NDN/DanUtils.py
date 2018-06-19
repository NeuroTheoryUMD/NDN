"""Neural deep network situtation-specific utils by Dan"""

from __future__ import division
import numpy as np
import NDN as NDN
#import NDN.NDNutils as NDNutils



def reg_path(
        NDNmodel=None,
        input_data=None,
        output_data=None,
        train_indxs=None,
        test_indxs=None,
        reg_type='l1',
        reg_vals=[1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1],
        ffnet_n=0,
        layer_n=0,
        data_filters=None,
        opt_params=None):

    """perform regularization over reg_vals to determine optimal cross-validated loss

        Args:

        Returns:
            dict: params to initialize an `FFNetwork` object

        Raises:
            TypeError: If `layer_sizes` is not specified
"""

    if NDNmodel is None:
        raise TypeError('Must specify NDN to regularize.')
    if input_data is None:
        raise TypeError('Must specify input_data.')
    if output_data is None:
        raise TypeError('Must specify output_data.')
    if train_indxs is None:
        raise TypeError('Must specify training indices.')
    if test_indxs is None:
        raise TypeError('Must specify testing indices.')

    num_regs = len(reg_vals)

    LLxs = np.zeros([num_regs],dtype='float32')
    test_mods = []

    for nn in range(num_regs):
        print( 'Regularization test:', reg_type,'=',reg_vals[nn] )
        test_mod = NDNmodel.copy_model()
        test_mod.set_regularization( reg_type, reg_vals[nn], ffnet_n, layer_n )
        test_mod.train(input_data=input_data, output_data=output_data,
                       train_indxs=train_indxs, test_indxs=test_indxs,
                       data_filters=data_filters, learning_alg='adam', opt_params=opt_params)
        LLxs[nn]=np.mean(
            test_mod.eval_models(input_data=input_data, output_data=output_data,
                                 data_indxs=test_indxs, data_filters=data_filters))
        test_mods.append( test_mod.copy_model() )
        print( nn, '(', reg_type, '=', reg_vals[nn], '): ', LLxs[nn])

    return LLxs, test_mods
# END reg_path
