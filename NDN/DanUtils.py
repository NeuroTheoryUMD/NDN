"""Neural deep network situtation-specific utils by Dan"""

from __future__ import division
import numpy as np
import NDN as NDN
import NDNutils as NDNutils
import matplotlib.pyplot as plt
from copy import deepcopy


def reg_path(
        ndn_mod=None,
        input_data=None,
        output_data=None,
        train_indxs=None,
        test_indxs=None,
        reg_type='l1',
        reg_vals=[1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1],
        ffnet_target=0,
        layer_target=0,
        data_filters=None,
        opt_params=None,
        fit_variables=None,
        output_dir=None):

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
        print('\nRegularization test: %s = %s:\n' % (reg_type, str(reg_vals[nn])))
        test_mod = ndn_mod.copy_model()
        test_mod.set_regularization(reg_type, reg_vals[nn], ffnet_target, layer_target)
        test_mod.train(input_data=input_data, output_data=output_data,
                       train_indxs=train_indxs, test_indxs=test_indxs,
                       data_filters=data_filters, fit_variables=fit_variables,
                       learning_alg='adam', opt_params=opt_params, output_dir=output_dir)
        LLxs[nn] = np.mean(
            test_mod.eval_models(input_data=input_data, output_data=output_data,
                                 data_indxs=test_indxs, data_filters=data_filters))
        test_mods.append(test_mod.copy_model())
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
    if test_indxs is None:
        raise TypeError('Must specify testing indices.')

    if data_filters is None:
        inds = test_indxs
        assert ndn_mod.filter_data is None, 'Must include data_filter given model history.'
    else:
        inds = np.intersect1d(test_indxs, np.where(data_filters[:, int(unit_number)] > 0))

    if len(inds) == 0:
        print("  Warning: no valid indices for cell %d." % unit_number)
        return 0

    all_LLs = ndn_mod.eval_models(
        input_data=input_data, output_data=output_data,
        data_indxs=inds, data_filters=data_filters, nulladjusted=False)

    # Need to cancel out and recalculate Poisson unit-norms, which might be based on
    # data_filtered firing rate (and not firing rate over inds)
    if (ndn_mod.noise_dist == 'poisson') and (ndn_mod.poisson_unit_norm is not None):
        real_norm = np.mean(output_data[inds, int(unit_number)])
        all_LLs = np.divide(
            np.multiply(all_LLs, ndn_mod.poisson_unit_norm[0]), real_norm)
        # note the zero indexing poisson norm is necessary because its now a list

    if not nulladjusted:
        LLreturn = all_LLs[int(unit_number)]
    else:
        LLreturn = -all_LLs[int(unit_number)]-ndn_mod.get_null_ll(output_data[inds, int(unit_number)])
 
    return LLreturn
# END filtered_eval_model


def spatial_profile_info(xprofile):
    """Calculate the mean and standard deviation of xprofile along one dimension"""
    # Calculate mean of filter

    if isinstance(xprofile, list):
        k = np.square(np.array(xprofile))
    else:
        k = np.square(xprofile.copy())

    NX = xprofile.shape[0]

    nrms = np.maximum(np.sum(k), 1e-10)
    mn_pos = np.divide(np.sum(np.multiply(k, range(NX))), nrms)
    xs = np.array([range(NX)] * np.ones([NX, 1])) - np.array([mn_pos] * np.ones([NX, 1]))
    stdev = np.sqrt(np.divide(np.sum(np.multiply(k, np.square(xs))), nrms))
    return mn_pos, stdev
# END spatial_profile_info


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

    # Check to see if there is a temporal layer first
    if np.prod(ndn_mod.networks[0].layers[0].filter_dims[1:]) == 1:
        ks = tbasis_recover_filters(ndn_mod)
        plt.plot(ndn_mod.networks[0].layers[0].weights)
        plt.title('Temporal basis')
        num_lags = ndn_mod.networks[0].layers[0].filter_dims[0]
        filter_width = ndn_mod.networks[0].layers[1].filter_dims[1]
    else:
        ks = ndn_mod.networks[0].layers[0].weights
        num_lags, filter_width = ndn_mod.networks[0].layers[0].filter_dims[:2]

    num_filters = ks.shape[1]

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
        ax = plt.subplot(rows, cols, nn + 1)
        plt.imshow(np.transpose(np.reshape(ks[:, nn], [filter_width, num_lags])),
                   cmap='Greys', interpolation='none',
                   vmin=-max(abs(ks[:, nn])), vmax=max(abs(ks[:, nn])), aspect='auto')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
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

    if plot_aspect != 'auto':
        plot_aspect = 'equal'

    # Check to see if NSM is convolutional or normal
    if (side_ndn.network_list[0]['layer_types'][0] == 'conv') or \
            (side_ndn.network_list[0]['layer_types'][0] == 'biconv'):
        is_conv = True
    else:
        is_conv = False

    # number of spatial dims depends on whether first side layer is convolutional or not
    #num_space = side_ndn.networks[0].input_dims[1]*side_ndn.networks[0].input_dims[2]
    num_space = int(side_ndn.networks[1].layers[0].weights.shape[0]/ side_ndn.networks[1].input_dims[0])
    num_cells = side_ndn.network_list[1]['layer_sizes'][-1]
    filter_nums = side_ndn.networks[1].num_units[:]
    #filter_nums = side_ndn.network_list[0]['layer_sizes'][:]
    num_layers = len(filter_nums)

    # Adjust effective space/filter number if binocular model
    if side_ndn.network_list[0]['layer_types'][0] == 'biconv':
        num_space = num_space // 2
        filter_nums[0] *= 2

    if cell_to_plot is not None:
        fig, ax = plt.subplots(nrows=1, ncols=num_layers)
        fig.set_size_inches(16, 3)

    # Reshape whole weight matrix
    wside = np.reshape(side_ndn.networks[1].layers[0].weights, [num_space, np.sum(filter_nums), num_cells])
    num_inh = side_ndn.network_list[0]['num_inh']

    # identify max and min weights for plotting (if plotting)
    if cell_to_plot is not None:
        if is_conv:
            # find normalization of layers for relevant cell
            img_max = np.max(wside[:, cell_to_plot])
            img_min = np.min(wside[:, cell_to_plot])
        else:
            img_max = np.max(wside)
            img_min = np.min(wside)
        # equalize scaling around zero
        if img_max > -img_min:
            img_min = -img_max
        else:
            img_max = -img_min

    fcount = 0
    ws = []
    for ll in range(num_layers):
        #wtemp = wside[range(ll, len(wside), num_layers), :]
        wtemp = wside[:, range(fcount, fcount+filter_nums[ll]), :]
        ws.append(wtemp.copy())
        fcount += filter_nums[ll]

        if cell_to_plot is not None:
            plt.subplot(1, num_layers, ll+1)
            if is_conv:
                plt.imshow(wtemp[:, :, cell_to_plot], aspect=plot_aspect, interpolation='none', cmap='bwr',
                           vmin=img_min, vmax=img_max)
                # Put line in for inhibitory units
                if num_inh[ll] > 0:
                    plt.plot(np.multiply([1, 1], filter_nums[ll]-num_inh[ll]-0.5), [-0.5, num_space-0.5], 'r')
                # Put line in for binocular layer (if applicable)
                if side_ndn.network_list[0]['layer_types'][ll] == 'biconv':
                    plt.plot([filter_nums[ll]/2, filter_nums[ll]/2], [-0.5, num_space-0.5], 'w')
            else:
                plt.imshow(np.transpose(wtemp), aspect='auto', interpolation='none', cmap='bwr',
                           vmin=img_min, vmax=img_max)  # will plot all cells
                # Put line in for inhibitory units
                if num_inh[ll] > 0:
                    plt.plot(np.multiply([1, 1], filter_nums[ll]-num_inh[ll]-0.5), [-0.5, num_cells-0.5], 'r')

    plt.show()
    return ws


def side_network_properties(side_ndn, norm_type=0):

    ws = side_network_analyze(side_ndn)
    wside = side_ndn.networks[-1].layers[-1].weights
    cell_nrms = np.sqrt(np.sum(np.square(wside), axis=0))
    NC = len(cell_nrms)
    num_layers = len(ws)
    NX = side_ndn.network_list[0]['input_dims'][1]

    if (side_ndn.network_list[0]['layer_types'][0] == 'conv') or (side_ndn.network_list[0]['layer_types'][0] == 'biconv'):
        conv_net = True
    else:
        conv_net = False
    assert conv_net is True, 'Convolutional network only for now.'

    # Calculate layer weights
    num_inh = side_ndn.network_list[0]['num_inh']
    layer_weights = np.zeros([num_layers, NC], dtype='float32')
    spatial_weights = np.zeros([num_layers, NX, NC], dtype='float32')
    EIlayer = np.zeros([2, num_layers, NC], dtype='float32')
    EIspatial = np.zeros([2, num_layers, NX, NC], dtype='float32')
    for ll in range(num_layers):
        if conv_net:
            layer_weights[ll, :] = np.sqrt(np.sum(np.sum(np.square(ws[ll]), axis=0), axis=0)) / cell_nrms
            spatial_weights[ll, :, :] = np.divide(np.sqrt(np.sum(np.square(ws[ll]), axis=1)), cell_nrms)
            if num_inh[ll] > 0:
                NE = ws[ll].shape[1] - num_inh[ll]
                elocs = range(NE)
                ilocs = range(NE, ws[ll].shape[1])
                if norm_type == 0:
                    EIspatial[0, ll, :, :] = np.sum(ws[ll][:, elocs, :], axis=1)
                    EIspatial[1, ll, :, :] = np.sum(ws[ll][:, ilocs, :], axis=1)
                    EIlayer[:, ll, :] = np.sum(EIspatial[:, ll, :, :], axis=1)
                else:
                    EIspatial[0, ll, :, :] = np.sqrt(np.sum(np.square(ws[ll][:, elocs, :]), axis=1))
                    EIspatial[1, ll, :, :] = np.sqrt(np.sum(np.square(ws[ll][:, ilocs, :]), axis=1))
                    EIlayer[:, ll, :] = np.sqrt(np.sum(np.square(EIspatial[:, ll, :, :]), axis=1))

        else:
            layer_weights[ll, :] = np.sqrt(np.sum(np.square(ws[ll]), axis=0)) / cell_nrms

    if np.sum(num_inh) > 0:
        if norm_type == 0:
            Enorm = np.sum(EIlayer[0, :, :], axis=0)
        else:
            Enorm = np.sqrt(np.sum(np.square(EIlayer[0, :, :]), axis=0))

        EIlayer = np.divide(EIlayer, Enorm)
        EIspatial = np.divide(EIspatial, Enorm)

    props = {'layer_weights': layer_weights,
             'spatial_profile': spatial_weights,
             'EIspatial': EIspatial,
             'EIlayer': EIlayer}

    return props


def side_distance(side_ndn, c1, c2, level=None, EI=None):
    """Assume network is convolutional -- otherwise wont make sense"""

    ws = side_network_analyze(side_ndn)
    NX, NC = ws[0].shape[0], ws[0].shape[2]
    assert (c1 < NC) and (c2 < NC), 'cells out of range'

    num_inh = side_ndn.network_list[0]['num_inh']
    NUs = side_ndn.network_list[0]['layer_sizes']
    if np.sum(num_inh) == 0:
        EI = None

    if level is not None:
        assert level < len(ws), 'level too large'
        if EI is not None:
            if EI > 0:  # then excitatory only
                w1 = ws[level][:, range(NUs[level]-num_inh[level]), c1]
                w2 = ws[level][:, range(NUs[level]-num_inh[level]), c2]
            else:  # then inhibitory only
                w1 = ws[level][:, range(NUs[level]-num_inh[level], NUs[level]), c1]
                w2 = ws[level][:, range(NUs[level]-num_inh[level], NUs[level]), c2]
        else:
            w1 = ws[level][:, :, c1]
            w2 = ws[level][:, :, c2]
    else:
        if EI is not None:
            if EI > 0:  # then excitatory only
                w1 = ws[0][:, range(NUs[0]-num_inh[0]), c1]
                w2 = ws[0][:, range(NUs[0]-num_inh[0]), c2]
                for ll in range(1, len(ws)):
                    w1 = np.concatenate((w1, ws[ll][:, range(NUs[ll]-num_inh[ll]), c1]), axis=1)
                    w2 = np.concatenate((w2, ws[ll][:, range(NUs[ll]-num_inh[ll]), c2]), axis=1)
            else:
                w1 = ws[0][:, range(NUs[0]-num_inh[0], NUs[0]), c1]
                w2 = ws[0][:, range(NUs[0]-num_inh[0], NUs[0]), c2]
                for ll in range(1, len(ws)):
                    w1 = np.concatenate((w1, ws[ll][:, range(NUs[ll]-num_inh[ll], NUs[ll]), c1]), axis=1)
                    w2 = np.concatenate((w2, ws[ll][:, range(NUs[ll]-num_inh[ll], NUs[ll]), c2]), axis=1)
        else:
            w1 = ws[0][:, :, c1]
            w2 = ws[0][:, :, c2]
            for ll in range(1, len(ws)):
                w1 = np.concatenate((w1, ws[ll][:, :, c1]), axis=1)
                w2 = np.concatenate((w2, ws[ll][:, :, c2]), axis=1)

    # Normalize
    nrm1 = np.sqrt(np.sum(np.square(w1)))
    nrm2 = np.sqrt(np.sum(np.square(w2)))
    if (nrm1 == 0) or (nrm2 == 0):
        return 0.0
    w1 = np.divide(w1, nrm1)
    w2 = np.divide(w2, nrm2)

    # Shift w2 to have highest overlap with w1
    ds = np.zeros(2*NX-1)
    for sh in range(2*NX-1):
        ds[sh] = np.sum(np.multiply(w1, NDNutils.shift_mat_zpad(w2, sh-NX+1, dim=0)))

    return np.max(ds)


def side_distance_vector(side_ndn, c1, level=None, EI=None):
    """compares distances between one cell and all others"""
    NC = side_ndn.networks[-1].layers[-1].weights.shape[1]
    dvec = np.zeros(NC, dtype='float32')
    for cc in range(NC):
        dvec[cc] = side_distance(side_ndn, c1, cc, level=level, EI=EI)
    dvec[c1] = 1

    return dvec


def side_distance_matrix(side_ndn, level=None, EI=None):

    NC = side_ndn.networks[-1].layers[-1].weights.shape[1]
    dmat = np.ones([NC, NC], dtype='float32')
    for c1 in range(NC):
        for c2 in range(c1+1,NC):
            dmat[c1, c2] = side_distance(side_ndn, c1, c2, level=level, EI=EI)
            dmat[c2, c1] = dmat[c1, c2]
    return dmat

def evaluate_ffnetwork(ffnet, end_weighting=None, to_plot=False, thresh_list=None, percent_drop=None):
    """Analyze FFnetwork nodes to determine their contribution in the big picture.
    thresh_list and percent_drop apply criteria for each layer (one or other) to suggest units to drop"""

    num_layers = len(ffnet.layers)
    num_unit_bot = ffnet.layers[-1].weights.shape[1]
    if end_weighting is None:
        prev_ws = np.ones(num_unit_bot, dtype='float32')
    else:
        assert len(end_weighting) == num_unit_bot, 'end_weighting has wrong dimensionality'
        prev_ws = end_weighting
    # Process prev_ws: nothing less than zeros, and sum to 1
    prev_ws = np.maximum(prev_ws, 0)
    prev_ws = np.divide(prev_ws, np.mean(prev_ws))

    node_eval = [[]]*num_layers
    node_eval[-1] = prev_ws.copy()
    for nn in range(num_layers-1):
        ws = ffnet.layers[num_layers-1-nn].weights.copy()
        next_ws = np.matmul(np.square(ws), prev_ws)
        node_eval[num_layers-nn-2] = np.divide(next_ws.copy(), np.mean(next_ws))
        prev_ws = next_ws.copy()

    # Determine units to drop (if any)
    units_to_drop = [[]]*num_layers
    remaining_units = [[]] * num_layers
    if percent_drop is not None:
        # overwrite thresh_list
        thresh_list = [0]*num_layers
        if len(percent_drop) == 1:
            percent_drop = percent_drop*num_layers
        for nn in range(num_layers):
            num_units = len(node_eval[nn])
            unit_order = np.argsort(node_eval[nn])
            cutoff = np.maximum(int(np.floor(num_units * percent_drop[nn])-1), 0)
            if cutoff == num_units - 1:
                remaining_units[nn] = range(num_units)
                thresh_list[nn] = 0
            else:
                remaining_units[nn] = unit_order[range(cutoff, num_units)]
                units_to_drop[nn] = unit_order[range(cutoff)]
                thresh_list[nn] = np.mean([node_eval[nn][unit_order[cutoff]], node_eval[nn][unit_order[cutoff + 1]]])
    else:
        if thresh_list is None:
            thresh_list = [None]*num_layers
        else:
            if thresh_list is not list:
                TypeError('thresh_list must be list.')
        if len(thresh_list) == 1:
            thresh_list = [thresh_list]*num_layers
        for nn in range(num_layers):
            if thresh_list[nn] is None:
                thresh_list[nn] = 0.2 * np.max(node_eval[nn])
            remaining_units[nn] = np.where(node_eval[nn] >= thresh_list[nn])[0]
            units_to_drop[nn] = np.where(node_eval[nn] < thresh_list[nn])[0]
            if len(remaining_units[nn]) == 0:
                print('layer %d: threshold too high' % nn)

    if to_plot:
        subplot_setup( num_rows=1, num_cols=num_layers)
        for nn in range(num_layers):
            plt.subplot(1, num_layers, nn+1)
            plt.plot(node_eval[nn], 'b')
            plt.plot(node_eval[nn], 'b.')
            NF = node_eval[nn].shape[0]
            plt.plot([0, NF-1], [thresh_list[nn], thresh_list[nn]], 'r')
            plt.xlim([0, NF-1])
        plt.show()

    return node_eval, remaining_units, units_to_drop


def tunnel_fit(ndn_mod, end_weighting=None, thresh_list=None, percent_drop=None):
    """Set up model with weights reset and coupled"""

    assert end_weighting is not None, 'Must supply end_weighting for this to work.'

    node_eval, good_nodes, tunnel_units = \
        evaluate_ffnetwork(ndn_mod.networks[0], end_weighting=end_weighting,
                           thresh_list=thresh_list, percent_drop=percent_drop)
    num_layers = len(node_eval)
    ndn_copy = ndn_mod.copy_model()

    # First randomize below-threshold filters in first level
    num_stix = ndn_copy.networks[0].layers[0].weights.shape[0]
    ndn_copy.networks[0].layers[0].weights[:, tunnel_units[0]] = \
        np.random.normal(size=[num_stix, len(tunnel_units[0])], scale=1/np.sqrt(num_stix))
    # Connect with rest of tunnel (and dissociate from rest of network
    for nn in range(1, num_layers):
        # Detach ok_units from previous-layer bad units
        #ok_units = list(set(range(len(node_eval[nn])))-set(tunnel_units[nn]))
        for mm in good_nodes[nn]:
            ndn_copy.networks[0].layers[nn].weights[tunnel_units[nn-1], mm] = 0
        for mm in tunnel_units[nn]:
            ndn_copy.networks[0].layers[nn].weights[:, mm] = np.zeros([len(node_eval[nn-1])], dtype='float32')
            ndn_copy.networks[0].layers[nn].weights[tunnel_units[nn-1], mm] = \
                np.random.normal(size=[len(tunnel_units[nn-1])], scale=1/np.sqrt(len(tunnel_units[nn-1])))

    return ndn_copy


def prune_ndn(ndn_mod, end_weighting=None, thresh_list=None, percent_drop=None):
    """Remove below-threshold nodes of network. Set thresholds to 0 if don't want to touch layer
        Also should not prune last layer (Robs), but can for multi-networks
        BUT CURRENTLY ONLY WORKS WITH SINGLE-NETWORK NDNs"""

    node_eval, remaining_units, _ = \
        evaluate_ffnetwork(ndn_mod.networks[0], end_weighting=end_weighting,
                           thresh_list=thresh_list, percent_drop=percent_drop)
    num_layers = len(node_eval)

    net_lists = deepcopy(ndn_mod.network_list)
    layer_sizes = net_lists[0]['layer_sizes']
    num_inh = net_lists[0]['num_inh']
    for nn in range(num_layers):
        if num_inh[nn] > 0:
            # update number of inhibitory units based on how many left
            num_inh[nn] = np.where(remaining_units[nn] > (layer_sizes[nn]-num_inh[nn]))[0].shape[0]
        layer_sizes[nn] = len(remaining_units[nn])

    net_lists[0]['layer_sizes'] = layer_sizes
    net_lists[0]['num_inh'] = num_inh

    # Make new NDN
    pruned_ndn = NDN.NDN(net_lists, noise_dist=ndn_mod.noise_dist, ffnet_out=ndn_mod.ffnet_out, tf_seed=ndn_mod.tf_seed)
    # Copy all the relevant weights and stuff
    for net_n in range(len(net_lists)):
        if net_n == 0:
            pruned_ndn.networks[0].layers[0].weights = \
                ndn_mod.networks[0].layers[0].weights[:, remaining_units[0]].copy()
            pruned_ndn.networks[0].layers[0].biases[0, :] = \
                ndn_mod.networks[0].layers[0].biases[0, remaining_units[0]].copy()
        else:
            pruned_ndn.networks[0].layers[0].weights = ndn_mod.networks[0].layers[0].weights.copy()
            pruned_ndn.networks[0].layers[0].biases = ndn_mod.networks[0].layers[0].biases.copy()

        for nn in range(1, len(pruned_ndn.networks[0].layers)):
            if net_n == 0:
                for mm in range(len(remaining_units[nn])):
                    cc = remaining_units[nn][mm]
                    pruned_ndn.networks[net_n].layers[nn].weights[:, mm] = \
                        ndn_mod.networks[net_n].layers[nn].weights[remaining_units[nn-1], cc].copy()
                    pruned_ndn.networks[net_n].layers[nn].biases[0, mm] = \
                        ndn_mod.networks[net_n].layers[nn].biases[0, cc]
            else:
                pruned_ndn.networks[net_n].layers[nn].weights = ndn_mod.networks[net_n].layers[nn].weights.copy()
                pruned_ndn.networks[net_n].layers[nn].biases = ndn_mod.networks[net_n].layers[nn].biases.copy()

    return pruned_ndn


def train_bottom_units( ndn_mod=None, unit_eval=None, num_units=None,
                        input_data=None, output_data=None, train_indxs=None, test_indxs=None,
                        data_filters=None, opt_params=None):

    MIN_UNITS = 10

    if ndn_mod is None:
        TypeError('Must define ndn_mod')
    if unit_eval is None:
        TypeError('Must input unit_eval')
    if input_data is None:
        TypeError('Forgot input_data')
    if output_data is None:
        TypeError('Forgot output_data')
    if train_indxs is None:
        TypeError('Forgot train_indxs')

    # Make new NDN
    netlist = deepcopy(ndn_mod.network_list)
    layer_sizes = netlist[0]['layer_sizes']
    num_units_full = layer_sizes[-1]

    # Default train bottom 20% of units
    if num_units is None:
        num_units = int(np.floor(num_units_full*0.2))
    size_ratio = num_units/num_units_full
    for nn in range(len(layer_sizes)-1):
        layer_sizes[nn] = int(np.maximum(size_ratio*layer_sizes[nn], MIN_UNITS))
        netlist[0]['num_inh'][nn] = int(np.floor(size_ratio*netlist[0]['num_inh'][nn]))
        netlist[0]['weights_initializers'][nn] = 'trunc_normal'
    layer_sizes[-1] = num_units
    netlist[0]['layer_sizes'] = layer_sizes  # might not be necessary because python is dumb
    netlist[0]['weights_initializers'][-1] = 'trunc_normal'
    small_ndn = NDN.NDN(netlist, noise_dist=ndn_mod.noise_dist)
    sorted_units = np.argsort(unit_eval)
    selected_units = sorted_units[range(num_units)]
    # Adjust Robs
    robs_small = output_data[:, selected_units]
    if data_filters is not None:
        data_filters_small = data_filters[:, selected_units]
    else:
        data_filters_small = None

    # Train
    _= small_ndn.train(input_data=input_data, output_data=robs_small, data_filters=data_filters_small,
                       learning_alg='adam', train_indxs=train_indxs, test_indxs=test_indxs, opt_params=opt_params)
    LLs = small_ndn.eval_models(input_data=input_data, output_data=robs_small,
                                data_indxs=test_indxs, data_filters=data_filters_small)

    return small_ndn, LLs, selected_units


def join_ndns(ndn1, ndn2, units2=None):
    """Puts all layers from both ndns into 1, except the last [output] layer, which is inherited
    from ndn1 only. However, new units of ndn2 (earlier layers will be connected """
    num_net = len(ndn1.networks)
    num_net2 = len(ndn2.networks)
    assert num_net == num_net2, 'Network number does not match'

    new_netlist = deepcopy(ndn1.network_list)
    num_units = [[]]*num_net
    for nn in range(num_net):
        num_layers = len(ndn1.networks[nn].layers)
        num_layers2 = len(ndn2.networks[nn].layers)
        assert num_layers == num_layers2, 'Layer number does not match'
        layer_sizes = [0]*num_layers
        num_units[nn] = [[]]*num_layers
        if ndn1.ffnet_out[0] == -1:
            ndn1.ffnet_out[0] = len(ndn1.networks)
        for ll in range(num_layers):
            if (nn != ndn1.ffnet_out[0]) or (ll < num_layers-1):
                num_units[nn][ll] = [ndn1.networks[nn].layers[ll].weights.shape[1],
                                     ndn2.networks[nn].layers[ll].weights.shape[1]]
            else:
                num_units[nn][ll] = [ndn1.networks[nn].layers[ll].weights.shape[1], 0]
            layer_sizes[ll] = np.sum(num_units[nn][ll])
            num_units[nn][ll][1] = layer_sizes[ll]  # need the sum anyway (see below)
            new_netlist[nn]['weights_initializers'][ll] = 'zeros'

        new_netlist[nn]['layer_sizes'] = layer_sizes

    joint_ndn = NDN.NDN(new_netlist, noise_dist=ndn1.noise_dist)
    # Assign weights and biases
    for nn in range(num_net):
        # First layer simple since input has not changed
        ll = 0
        joint_ndn.networks[nn].layers[ll].weights[:, range(num_units[nn][ll][0])] = \
            ndn1.networks[nn].layers[ll].weights
        joint_ndn.networks[nn].layers[ll].weights[:, range(num_units[nn][ll][0], num_units[nn][ll][1])] = \
            ndn2.networks[nn].layers[ll].weights
        joint_ndn.networks[nn].layers[ll].biases[:, range(num_units[nn][ll][0])] = \
            ndn1.networks[nn].layers[ll].biases
        joint_ndn.networks[nn].layers[ll].biases[:, range(num_units[nn][ll][0], num_units[nn][ll][1])] = \
            ndn2.networks[nn].layers[ll].biases

        for ll in range(1, len(num_units[nn])):
            joint_ndn.networks[nn].layers[ll].biases[:, range(num_units[nn][ll][0])] =\
                ndn1.networks[nn].layers[ll].biases.copy()
            weight_strip = np.zeros([num_units[nn][ll-1][1], num_units[nn][ll][0]], dtype='float32')
            weight_strip[range(num_units[nn][ll-1][0]), :] = ndn1.networks[nn].layers[ll].weights.copy()
            joint_ndn.networks[nn].layers[ll].weights[:, range(num_units[nn][ll][0])] = weight_strip
            if (nn != ndn1.ffnet_out[0]) or (ll < num_layers-1):
                weight_strip = np.zeros([num_units[nn][ll - 1][1],
                                         num_units[nn][ll][1] - num_units[nn][ll][0]], dtype='float32')
                weight_strip[range(num_units[nn][ll - 1][0], num_units[nn][ll - 1][1]), :] = \
                    ndn2.networks[nn].layers[ll].weights.copy()
                joint_ndn.networks[nn].layers[ll].weights[:, range(num_units[nn][ll][0], num_units[nn][ll][1])] =\
                    weight_strip
                joint_ndn.networks[nn].layers[ll].biases[:, range(num_units[nn][ll][0], num_units[nn][ll][1])] =\
                    ndn2.networks[nn].layers[ll].biases.copy()
            elif units2 is not None:
                weight_strip = np.zeros([num_units[nn][ll - 1][1], len(units2)], dtype='float32')
                weight_strip[range(num_units[nn][ll - 1][0], num_units[nn][ll - 1][1]), :] = \
                    ndn2.networks[nn].layers[ll].weights.copy()
                joint_ndn.networks[nn].layers[ll].weights[:, units2] = weight_strip

    return joint_ndn


def subplot_setup(num_rows, num_cols, row_height=2):
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols)
    fig.set_size_inches(16, row_height*num_rows)


def matlab_export(filename, variable_list):
    """Export list of variables to .mat file"""

    import scipy.io as sio
    if not isinstance(variable_list, list):
        variable_list = [variable_list]

    matdata = {}
    for nn in range(len(variable_list)):
        assert not isinstance(variable_list[nn], list), 'Cant do a list of lists.'
        if nn < 10:
            key_name = 'v0' + str(nn)
        else:
            key_name = 'v' + str(nn)
        matdata[key_name] = variable_list[nn]

    sio.savemat(filename, matdata)


def tbasis_recover_filters(ndn_mod):

    assert np.prod(ndn_mod.networks[0].layers[0].filter_dims[1:]) == 1, 'only works with temporal-only basis'

    idims = ndn_mod.networks[0].layers[1].filter_dims
    nlags = ndn_mod.networks[0].layers[0].filter_dims[0]
    num_filts = ndn_mod.networks[0].layers[1].weights.shape[1]
    tkerns = ndn_mod.networks[0].layers[0].weights

    ks = np.zeros([idims[1]*idims[2]*nlags, num_filts])
    for nn in range(num_filts):
        w = np.reshape(ndn_mod.networks[0].layers[1].weights[:, nn], [idims[2], idims[1], idims[0]])
        k = np.zeros([idims[2], idims[1], nlags])
        for yy in range(idims[2]):
            for xx in range(idims[1]):
                k[yy, xx, :] = np.matmul(tkerns, w[yy, xx, :])
        ks[:, nn] = np.reshape(k, idims[1]*idims[2]*nlags)

    return ks

def side_ei_analyze( side_ndn ):

    num_space = int(side_ndn.networks[0].input_dims[1])
    num_cells = side_ndn.network_list[1]['layer_sizes'][-1]

    num_units = side_ndn.networks[1].num_units[:]
    num_layers = len(num_units)

    wside = np.reshape(side_ndn.networks[1].layers[0].weights, [num_space, np.sum(num_units), num_cells])
    num_inh = side_ndn.network_list[0]['num_inh']

    cell_nrms = np.sum(side_ndn.networks[1].layers[0].weights, axis=0)
    tlayer_present = side_ndn.networks[0].layers[0].filter_dims[1] == 1
    if tlayer_present:
        nlayers = num_layers - 1
    else:
        nlayers = num_layers
    # EIweights = np.zeros([2, nlayers, num_cells])
    EIprofiles = np.zeros([2, nlayers, num_space, num_cells])
    num_exc = np.subtract(num_units, num_inh)

    fcount = 0
    for ll in range(num_layers):
        ws = wside[:, range(fcount, fcount+num_units[ll]), :].copy()
        fcount += num_units[ll]
        if num_inh[ll] == 0:
            ews = np.maximum(ws, 0)
            iws = np.minimum(ws, 0)
        else:
            ews = ws[:, range(num_exc[ll]), :]
            iws = ws[:, range(num_exc[ll], num_units[ll]), :]
        if (ll == 0) or (tlayer_present == False):
            EIprofiles[0, ll, :, :] = np.divide( np.sum(ews, axis=1), cell_nrms )
            EIprofiles[1, ll, :, :] = np.divide( np.sum(iws, axis=1), cell_nrms )
        if tlayer_present:
            EIprofiles[0, ll-1, :, :] += np.divide( np.sum(ews, axis=1), cell_nrms )
            EIprofiles[1, ll-1, :, :] += np.divide( np.sum(iws, axis=1), cell_nrms )
    EIweights = np.sum(EIprofiles, axis=2)

    return EIweights, EIprofiles


def scaffold_nonconv_plot( side_ndn, with_inh=True, nolabels=True, skip_first_level=False, linewidth=1):

    # validity check
    assert len(side_ndn.network_list) == 2, 'This does not seem to be a standard scaffold network.'

    num_cells = side_ndn.network_list[1]['layer_sizes'][-1]
    num_units = side_ndn.networks[1].num_units[:]
    num_layers = len(num_units)
    scaff_ws = side_ndn.networks[1].layers[0].weights
    cell_nrms = np.max(np.abs(side_ndn.networks[1].layers[0].weights), axis=0)
    num_inh = side_ndn.network_list[0]['num_inh']
    num_exc = np.subtract(num_units, num_inh)

    fcount = 0
    col_mod = 0
    if skip_first_level:
        col_mod = 1

    subplot_setup(num_rows=1, num_cols=num_layers-col_mod)
    plt.rcParams['lines.linewidth'] = linewidth
    plt.rcParams['axes.linewidth'] = linewidth
    for ll in range(num_layers):
        ws = np.transpose(np.divide(scaff_ws[range(fcount, fcount+num_units[ll]), :].copy(), cell_nrms))

        fcount += num_units[ll]
        if (num_inh[ll] > 0) and with_inh:
            ws[:, num_exc[ll]:] = np.multiply( ws[:, num_exc[ll]:], -1)
        if not skip_first_level or (ll > 0):
            ax = plt.subplot(1, num_layers-col_mod, ll+1-col_mod)

            if with_inh:
                plt.imshow(ws, aspect='auto', interpolation='none', cmap='bwr', vmin=-1, vmax=1)
            else:
                plt.imshow(ws, aspect='auto', interpolation='none', cmap='Greys', vmin=0, vmax=1)
            if num_inh[ll] > 0:
                plt.plot(np.multiply([1, 1], num_exc[ll]-0.5), [-0.5, num_cells-0.5], 'k')
            if ~nolabels:
                ax.set_xticks([])
                ax.set_yticks([])
    plt.show()


def scaffold_plot_cell( side_ndn, cell_n, with_inh=True, nolabels=True, skip_first_level=False, linewidth=1):

    # validity check
    assert len(side_ndn.network_list) == 2, 'This does not seem to be a standard scaffold network.'

    num_cells = side_ndn.network_list[1]['layer_sizes'][-1]
    num_units = side_ndn.networks[1].num_units[:]
    num_layers = len(num_units)

    if 'biconv' in side_ndn.network_list[0]['layer_types']:
        num_space = int(side_ndn.networks[0].input_dims[1])//2
    else:
        num_space = int(side_ndn.networks[0].input_dims[1])

    cell_nrms = np.max(np.abs(side_ndn.networks[1].layers[0].weights), axis=0)
    wside = np.reshape(side_ndn.networks[1].layers[0].weights, [num_space, np.sum(num_units), num_cells])
    num_inh = side_ndn.network_list[0]['num_inh']
    num_exc = np.subtract(num_units, num_inh)
    #cell_nrms = np.sum(side_ndn.networks[1].layers[0].weights, axis=0)

    fcount = 0
    col_mod = 0
    if skip_first_level:
        col_mod = 1

    subplot_setup(num_rows=1, num_cols=num_layers-col_mod)
    plt.rcParams['lines.linewidth'] = linewidth
    plt.rcParams['axes.linewidth'] = linewidth
    for ll in range(num_layers):
        ws = np.divide(wside[:, range(fcount, fcount+num_units[ll]), cell_n].copy(), cell_nrms[cell_n])

        fcount += num_units[ll]
        if (num_inh[ll] > 0) and with_inh:
            ws[:, num_exc[ll]:] = np.multiply( ws[:, num_exc[ll]:], -1)
            if side_ndn.network_list[0]['layer_types'][ll] == 'biconv':
                more_inh_range = range(num_units[ll]//2-num_inh[ll], num_units[ll]//2)
                ws[:, more_inh_range] = np.multiply(ws[:, more_inh_range], -1)
        if not skip_first_level or (ll > 0):
            ax = plt.subplot(1, num_layers-col_mod, ll+1-col_mod)

            if with_inh:
                plt.imshow(ws, aspect='auto', interpolation='none', cmap='bwr', vmin=-1, vmax=1)
            else:
                plt.imshow(ws, aspect='auto', interpolation='none', cmap='Greys', vmin=0, vmax=1)
            if side_ndn.network_list[0]['layer_types'][ll] == 'biconv':
                plt.plot(np.multiply([1, 1], num_units[ll]//2 - 0.5), [-0.5, num_space - 0.5], 'k')
            if num_inh[ll] > 0:
                plt.plot(np.multiply([1, 1], num_exc[ll]-0.5), [-0.5, num_space-0.5], 'k')
                if side_ndn.network_list[0]['layer_types'][ll] == 'biconv':
                    plt.plot(np.multiply([1, 1], num_exc[ll]-num_units[ll]//2 - 0.5), [-0.5, num_space - 0.5], 'k')
            if ~nolabels:
                ax.set_xticks([])
                ax.set_yticks([])
    plt.show()


def plot_2dweights( w, input_dims=None, num_inh=0):
    """w can be one dimension (in which case input_dims is required) or already shaped. """

    if input_dims is not None:
        w = np.reshape(w, [input_dims[1], input_dims[0]])
    else:
        input_dims = [w.shape[1], w.shape[0]]

    num_exc = input_dims[0]-num_inh
    w = np.divide(w, np.max(np.abs(w)))

    fig, ax = plt.subplots(1)

    if num_inh > 0:
        w[:, num_exc:] = np.multiply(w[:, num_exc:], -1)
        plt.imshow(w, interpolation='none', cmap='bwr', vmin=-1, vmax=1)
    else:
        plt.imshow(w, aspect='auto', interpolation='none', cmap='Greys', vmin=0, vmax=1)
    if num_inh > 0:
        plt.plot(np.multiply([1, 1], num_exc-0.5), [-0.5, input_dims[1]-0.5], 'k')
    ax.set_yticklabels([])
    ax.set_xticklabels([])


def entropy(dist):

    # normalize distribution
    dist = np.divide( dist.astype('float32'), np.sum(dist) )
    # make all zeros 1
    dist[np.where(dist == 0)[0]] = 1
    H = -np.sum( np.multiply( dist, np.log2(dist)) )

    return H


def best_val_mat(mat, min_or_max=0):

    # Make sure a matrix (numpy)
    mat = np.array(mat, dtype='float32')
    if min_or_max == 0:
        ax0 = np.min(mat, axis=0)
        b1 = np.argmin(ax0)
        b0 = np.argmin(mat[:, b1])
    else:
        ax0 = np.max(mat, axis=0)
        b1 = np.argmax(ax0)
        b0 = np.argmax(mat[:, b1])

    b0 = int(b0)
    b1 = int(b1)
    return b0, b1


def ffnet_health(ndn_mod, toplot=True):

    num_nets = len(ndn_mod.networks)
    whealth = [None]*num_nets
    bhealth = [None]*num_nets
    max_layers = 0
    for nn in range(num_nets):
        num_layers = len(ndn_mod.networks[nn].layers)
        whealth[nn] = [None]*num_layers
        bhealth[nn] = [None]*num_layers
        if num_layers > max_layers:
            max_layers = num_layers
        for ll in range(num_layers):
            whealth[nn][ll] = np.std(ndn_mod.networks[nn].layers[ll].weights, axis=0)
            bhealth[nn][ll] = ndn_mod.networks[nn].layers[ll].biases[0, :]

    if toplot:
        subplot_setup(num_nets * 2, max_layers + 1)
        for nn in range(num_nets):
            num_layers = len(ndn_mod.networks[nn].layers)
            for ll in range(num_layers):
                plt.subplot(num_nets*2, max_layers, (2*nn)*max_layers+ll+1)
                _=plt.hist(whealth[nn][ll], 20)
                plt.title("n%dL%d ws" % (nn, ll))
                plt.subplot(num_nets*2, max_layers, (2*nn+1)*max_layers+ll+1)
                _=plt.hist(bhealth[nn][ll], 20)
                plt.title("n%dL%d bs" % (nn, ll))
        plt.show()

    return whealth, bhealth

