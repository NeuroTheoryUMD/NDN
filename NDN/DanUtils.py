"""Neural deep network situtation-specific utils by Dan"""

from __future__ import division
import numpy as np
#import NDN as NDN
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
        opt_params=None,
        variable_list=None):

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
                       data_filters=data_filters, variable_list=variable_list,
                       learning_alg='adam', opt_params=opt_params)
        LLxs[nn] = np.mean(
            test_mod.eval_models(input_data=input_data, output_data=output_data,
                                 data_indxs=test_indxs, data_filters=data_filters))
        test_mods.append( test_mod.copy_model() )
        print( nn, '(', reg_type, '=', reg_vals[nn], '): ', LLxs[nn])

    return LLxs, test_mods
# END reg_path


def filtered_eval_model(
        unit_number,
        NDNmodel=None,
        input_data=None,
        output_data=None,
        test_indxs=None,
        data_filters=None,
        nulladjusted=False):

    """This will return each neuron model evaluated on valid indices (given datafilter).
    It will also return those valid indices for each unit"""

    if NDNmodel is None:
        raise TypeError('Must specify NDN to regularize.')
    if input_data is None:
        raise TypeError('Must specify input_data.')
    if output_data is None:
        raise TypeError('Must specify output_data.')
    if data_filters is None:
        raise TypeError('Must specify data_filters.')
    if test_indxs is None:
        raise TypeError('Must specify testing indices.')

    #NT, NU = data_filters.shape
    #LLx = np.zeros([NU], dtype='float32')
    #XVindx_list = []
    #for cc in range(NU):
    inds = np.intersect1d(test_indxs, np.where(data_filters[:, int(unit_number)] > 0))
    # need to make sure normalized by neuron's own firing rate
    FRchoice = NDNmodel.poisson_unit_norm
    NDNmodel.poisson_unit_norm = True
    all_LLs = NDNmodel.eval_models(
        input_data=input_data, output_data=output_data,
        data_indxs=inds, data_filters=data_filters, nulladjusted=False)
    if nulladjusted == False:
        LLreturn = all_LLs[int(unit_number)]
    else:
        LLreturn = -all_LLs[int(unit_number)]-NDNmodel.nullLL(output_data[inds, int(unit_number)])
    # turn back to original value
    NDNmodel.poisson_unit_norm = FRchoice

    return LLreturn
# END filtered_eval_model


def spatial_spread(filters, axis=0):
    """Calculate the spatial spread of a list of filters along one dimension"""
    # Calculate mean of filter
    k = np.square(filters.copy())
    if axis > 0:
        k = np.transpose(k)
    NX, NF = filters.shape

    nrms = np.maximum(np.sum(k,axis=0), 1e-10)
    mn_pos = np.divide(np.sum(np.multiply(np.transpose(k), range(NX)), axis=1), nrms)
    xs = np.array([range(NX)] * np.ones([NF, 1])) - np.transpose(np.array([mn_pos] * np.ones([NX, 1])))
    stdevs = np.sqrt(np.divide(np.sum(np.multiply(np.transpose(k), np.square(xs)), axis=1), nrms))

    return stdevs
# END spatial_spread
