"""Neural deep network situtation-specific utils by Dan"""

from __future__ import division
import numpy as np
#import NDN as NDN
#import NDN.NDNutils as NDNutils


def reg_path(
        ndn_mod=None,
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

    if ndn_mod is None:
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
        print('\nRegulariation test: %s = %s:\n' % (reg_type, str(reg_vals[nn])))
        test_mod = ndn_mod.copy_model()
        test_mod.set_regularization(reg_type, reg_vals[nn], ffnet_n, layer_n)
        test_mod.train(input_data=input_data, output_data=output_data,
                       train_indxs=train_indxs, test_indxs=test_indxs,
                       data_filters=data_filters, variable_list=variable_list,
                       learning_alg='adam', opt_params=opt_params)
        LLxs[nn] = np.mean(
            test_mod.eval_models(input_data=input_data, output_data=output_data,
                                 data_indxs=test_indxs, data_filters=data_filters))
        test_mods.append( test_mod.copy_model() )
        print('%s (%s = %s): %s' % (nn, reg_type, reg_vals[nn], LLxs[nn]))

    return LLxs, test_mods
# END reg_path


def safe_generate_predictions(
        ndn_model=None,
        input_data=None,
        data_indxs=None,
        output_units=None,
        safe_blk_size=100000):

    """This will return each neuron model evaluated on valid indices (given datafilter).
    It will also return those valid indices for each unit"""

    if ndn_model is None:
        raise TypeError('Must specify NDN to regularize.')
    if input_data is None:
        raise TypeError('Must specify input_data.')
    if data_indxs is None:
        X = input_data
    else:
        assert np.max(data_indxs) < input_data.shape[0], 'data_indxs too large'
        X = input_data[data_indxs, :]
    NT = X.shape[0]
    num_outputs = np.prod(ndn_model.networks[ndn_model.ffnet_out[0]].layers[-1].output_dims)
    if output_units is None:
        output_units = range(num_outputs)
    else:
        assert np.max(output_units) < num_outputs

    preds = np.zeros([NT, num_outputs])

    for nn in range(int(np.ceil(NT/safe_blk_size))):
        indxs = range(safe_blk_size*nn, np.min([NT, safe_blk_size*(nn+1)]))
        print('Generating prediction for', indxs[0], 'to', indxs[-1])
        tmp = ndn_model.generate_prediction(input_data=X[indxs, :])
        preds[indxs, :] = tmp[:, output_units]

    return preds
# END safe_generate_predictions


def filtered_eval_model(
        unit_number,
        ndn_mod=None,
        input_data=None,
        output_data=None,
        test_indxs=None,
        data_filters=None,
        nulladjusted=False):

    """This will return each neuron model evaluated on valid indices (given datafilter).
    It will also return those valid indices for each unit"""

    if ndn_mod is None:
        raise TypeError('Must specify NDN to regularize.')
    if input_data is None:
        raise TypeError('Must specify input_data.')
    if output_data is None:
        raise TypeError('Must specify output_data.')
    if data_filters is None:
        raise TypeError('Must specify data_filters.')
    if test_indxs is None:
        raise TypeError('Must specify testing indices.')

    inds = np.intersect1d(test_indxs, np.where(data_filters[:, int(unit_number)] > 0))
    # need to make sure normalized by neuron's own firing rate
    FRchoice = ndn_mod.poisson_unit_norm
    ndn_mod.poisson_unit_norm = True
    all_LLs = ndn_mod.eval_models(
        input_data=input_data, output_data=output_data,
        data_indxs=inds, data_filters=data_filters, nulladjusted=False)
    if not nulladjusted:
        LLreturn = all_LLs[int(unit_number)]
    else:
        LLreturn = -all_LLs[int(unit_number)]-ndn_mod.nullLL(output_data[inds, int(unit_number)])
    # turn back to original value
    ndn_mod.poisson_unit_norm = FRchoice

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


def plot_filters(ndn_mod):

    import matplotlib.pyplot as plt  # plotting

    ks = ndn_mod.networks[0].layers[0].weights
    num_filters = ks.shape[1]
    num_lags = ndn_mod.network_list[0]['input_dims'][0]
    filter_width = ks.shape[0] // num_lags
    
    if num_filters/10 == num_filters//10:
        cols = 10
    elif num_filters / 8 == num_filters // 8:
        cols = 8
    elif num_filters / 6 == num_filters // 6:
        cols = 6
    elif num_filters / 5 == num_filters // 5:
        cols = 5
    else:
        cols = 8
    rows = int(np.ceil(num_filters/cols))

    fig, ax = plt.subplots(nrows=rows, ncols=cols)
    fig.set_size_inches(18 / 6 * cols, 7 / 4 * rows)
    for nn in range(num_filters):
        plt.subplot(rows, cols, nn + 1)
        plt.imshow(np.transpose(np.reshape(ks[:, nn], [filter_width, num_lags])),
                   cmap='Greys', interpolation='none',
                   vmin=-max(abs(ks[:, nn])), vmax=max(abs(ks[:, nn])), aspect='auto')
    plt.show()
# END plot_filters


def side_network_analyze(side_ndn, cell_to_plot=None, plot_aspect='auto'):
    """
    Applies to NDN with a side network (conv or non-conv. It will divide up the weights
    of the side-network into layer-specific pieces and resize to represent space and
    filter number as different inputs.

    Inputs:
        side_ndn: the network model (required)
        cell_to_plot: if plots desired, single out the cell to plot (default no plot). If
            convolutional, then can only plot specified cell. If non-convolutonal, then
            just set to something other than 'None', and will plot weights for all cells.
        plot_aspect: if plots, then whether to have aspect as 'auto' (default) or 'equal'
    Output:
        returns the weights as organized as descrived above
    """
    import matplotlib.pyplot as plt  # plotting
    if plot_aspect != 'auto':
        plot_aspect = 'equal'

    # Check to see if NSM is convolutional or normal
    if (side_ndn.network_list[0]['layer_types'][0] == 'conv') or \
            (side_ndn.network_list[0]['layer_types'][0] == 'biconv'):
        is_conv = True
    else:
        is_conv = False

    num_space = side_ndn.network_list[0]['input_dims'][1]
    num_cells = side_ndn.network_list[1]['layer_sizes'][-1]
    filter_nums = side_ndn.network_list[0]['layer_sizes'][:]
    num_layers = len(filter_nums)

    # Adjust effective space/filter number if binocular model
    if side_ndn.network_list[0]['layer_types'][0] == 'biconv':
        num_space = num_space // 2
        filter_nums[0] *= 2

    if cell_to_plot is not None:
        fig, ax = plt.subplots(nrows=1, ncols=num_layers)
        fig.set_size_inches(16, 3)

    wside = side_ndn.networks[1].layers[0].weights
    num_inh = side_ndn.network_list[0]['num_inh']
    ws = []
    for ll in range(num_layers):
        wtemp = wside[range(ll, len(wside), num_layers), :]
        if is_conv:
            ws.append(np.reshape(wtemp[range(filter_nums[ll] * num_space), :],
                                 [num_space, filter_nums[ll], num_cells]))
        else:
            ws.append(np.reshape(wtemp[range(filter_nums[ll]), :], [filter_nums[ll], num_cells]))

        if cell_to_plot is not None:
            plt.subplot(1, num_layers, ll+1)
            if is_conv:
                plt.imshow(ws[ll][:, :, cell_to_plot], aspect=plot_aspect)
                # Put line in for inhibitory units
                if num_inh[ll] > 0:
                    plt.plot(np.multiply([1, 1], filter_nums[ll]-num_inh[ll]-0.5), [-0.5, num_space-0.5], 'r')
                # Put line in for binocular layer (if applicable)
                if side_ndn.network_list[0]['layer_types'][ll] == 'biconv':
                    plt.plot([filter_nums[ll]/2, filter_nums[ll]/2], [-0.5, num_space-0.5], 'w')
            else:
                plt.imshow(np.transpose(ws[ll]), aspect='auto')  # will plot all cells
                # Put line in for inhibitory units
                if num_inh[ll] > 0:
                    plt.plot(np.multiply([1, 1], filter_nums[ll]-num_inh[ll]-0.5), [-0.5, num_cells-0.5], 'r')

    plt.show()

    return ws




