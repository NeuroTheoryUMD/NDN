"""Neural deep network situtation-specific utils by Hadi"""

from __future__ import print_function
from __future__ import division

import os
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import math
import datetime
from matplotlib.backends.backend_pdf import PdfPages
from prettytable import PrettyTable


def r_squared(true, pred, data_indxs=None):
    """
    START.

    :param true: vector containing true values
    :param pred: vector containing predicted (modeled) values
    :param data_indxs: obv.
    :return: R^2

    It is assumed that vectors are organized in columns

    END.
    """

    assert true.shape == pred.shape, 'true and prediction vectors should have the same shape'

    if data_indxs is None:
        dim = true.shape[0]
        data_indxs = np.arange(dim)
    else:
        dim = len(data_indxs)

    ss_res = np.sum(np.square(true[data_indxs, :] - pred[data_indxs, :]), axis=0) / dim
    ss_tot = np.var(true[data_indxs, :], axis=0)

    return 1 - ss_res/ss_tot


def crop(input_k, input_dims, x_range, y_range):
    """
    START.

    :param input_k: input array. vectors are arranged in columns and each have input_k.shape[0] elements
    :param input_dims: [nlags, NX, NY]
    :param x_range: obv
    :param y_range: obv
    :return: cropped version

    END.
    """

    x0, x1 = min(x_range), max(x_range)
    y0, y1 = min(y_range), max(y_range)
    assert ((x0 and x1) <= input_dims[1]) and ((x0 and x1) >= 0), 'must be in range.'
    assert ((y0 and y1) <= input_dims[2]) and ((y0 and y1) >= 0), 'must be in range.'

    interval = []
    for y in range(y0, y1):
        interval.extend(range(x0*input_dims[0] + y*input_dims[0]*input_dims[1],
                              (x1-1)*input_dims[0] + y*input_dims[0]*input_dims[1] + input_dims[0]))

    out = input_k[interval, :]
    return out


def get_gaussian_filter(spatial_dims, sigmas_dict, centers_dict=None, alpha=0.4,
                        normalize=True, plot=True):
    """
    :param spatial_dims: [width_x, width_y]
    :param sigmas_dict: e.g. {'sigma_0': [2, 0], 'sigma_1': [2, 3]}
    :param centers_dict: same as sigmas
    :param alpha: a parameter for center-surround-ness (alpha --> 0: no surround)
    :param normalize: Obv.
    :param plot: Obv.
    :return:
    """

    # problem dims
    width_x, width_y = spatial_dims[0], spatial_dims[1]
    sigmas = sorted(sigmas_dict.items())
    filts_n = len(sigmas)

    for mu in range(filts_n):
        assert sigmas[mu][1][0] > 0, 'all sigma[0]s should be greater than zero.'

    # make the fitlers
    filts = np.zeros((width_x * width_y, filts_n), dtype='float32')

    for mu in range(filts_n):
        sigma1 = sigmas[mu][1][0]
        sigma2 = sigmas[mu][1][1]

        if centers_dict is None:
            center_x = int(width_x // 2)
            center_y = int(width_y // 2)
        else:
            center_x = sorted(centers_dict.items())[mu][1][0]
            center_y = sorted(centers_dict.items())[mu][1][1]

        for i in range(width_x * width_y):
            x_pos = (i % width_x)
            y_pos = i // width_x

            # this is the radius corresponding to i
            r = np.sqrt((x_pos - center_x) ** 2 + (y_pos - center_y) ** 2)

            gauss1 = math.exp(-np.square(r) / (2 * np.square(sigma1))) /\
                     (2 * math.pi * np.square(sigma1))
            filts[i, mu] = gauss1

            if sigma2 > 0:
                gauss2 = math.exp(-np.square(r) / (2 * np.square(sigma2))) /\
                         (2 * math.pi * np.square(sigma2))
                filts[i, mu] = gauss1 - alpha * np.square((sigma2 / sigma1)) * gauss2

    if normalize:
        filts /= np.linalg.norm(filts, axis=0)

    if plot:
        plt.figure(figsize=(4*filts_n, 1 + filts_n//2))

        for mu in range(filts_n):
            argmax = np.argmax(filts[:, mu])
            x0_pos = argmax % width_x
            y0_pos = argmax // width_x

            plt.subplot(1, filts_n, mu+1)
            k = np.reshape(filts[:, mu], (width_y, width_x))
            plt.imshow(k, cmap='Greys', interpolation=None,
                       vmin=-max(abs(k.flatten())), vmax=max(abs(k.flatten())))
            plt.title('Filter # %s, centered at: %s, %s' % (mu, x0_pos, y0_pos))
            plt.colorbar()
        plt.show()

    return filts


def space_embedding(k, dims, x_rng, y_rng):
    """
    :param k: vector with first dim as space
    :param dims: spatial dims of the larger space [nx, ny]
    :param x_rng: x range of the embedding window
    :param y_rng: y range of the embedding window
    :return: k embedded in the larger space
    """

    assert len(dims) == 2, 'dims should be a list of two integers: [NX, NY].'
    assert len(x_rng) == 2, 'x_rng should be a list of two integers.'
    assert len(y_rng) == 2, 'x_rng should be a list of two integers.'

    nx, ny = sorted(dims)
    x_start, x_end = sorted(x_rng)
    y_start, y_end = sorted(y_rng)

    delta_x, delta_y = x_end - x_start, y_end - y_start
    assert delta_y * delta_x == k.shape[0], 'k should be the same size as the window provided.'
    other_dim = k.shape[1]

    x_start_small, x_end_small = 0, delta_x

    if x_start < 0:
        x_start_small = -x_start
        x_start = 0
    elif x_end > nx:
        x_end_small = delta_x - (x_end - nx)
        x_end = nx

    k_embedded = np.zeros((np.prod(dims), other_dim))

    for yy in range(y_start, y_end):
        if yy < 0 or yy >= ny:
            continue
        big_intvl = range(yy * dims[0] + x_start,
                          yy * dims[0] + x_end)
        small_intvl = range((yy - y_start) * delta_x + x_start_small,
                            (yy - y_start) * delta_x + x_end_small)
        k_embedded[big_intvl, :] = k[small_intvl, :]

    return k_embedded


def subunit_plots(ndn, mode='kers', layer=1, sub_indxs=None, only_sep_plot=True,
                  fig_sz_sep_s=(20, 20), fig_sz_sep_t=(40, 20), fig_sz_nonsep=(20, 12),
                  save_dir='./plots/'):
    #_allowed_modes = ['tbasis', 'sbasis', 'kers', 'subs', 'neurons']

    print('Plotting... mode: --%s--, layer: --%s--' % (mode, layer))

    if not os.path.isdir(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))

    tbasis = ndn.networks[0].layers[0].weights
    nlags, tbasis_n = tbasis.shape
    [_, width_x, width_y] = ndn.networks[0].layers[1].filter_dims
    kers = deepcopy(ndn.networks[0].layers[1].weights)

    # TODO: if mod_width > 1 and layer >= 2 then generate subs in a different way
    subs = deepcopy(kers)
    for ii in range(layer - 1):
        if mode == 'cells' and ii == layer-2:
            mods_n = ndn.networks[0].layers[layer].input_dims[0]
            w_readout_filt = deepcopy(ndn.networks[0].layers[layer].weights[:mods_n, :])
            subs = np.matmul(subs, w_readout_filt)
        else:
            subs = np.matmul(subs, ndn.networks[0].layers[ii + 2].weights)

    subs_n = ndn.networks[0].layers[layer].num_filters

    d, h, m = datetime.datetime.now().day, datetime.datetime.now().hour, datetime.datetime.now().minute
    file_name = '%s,%s:%s_%s.pdf' % (d, h, m, mode)
    pp = PdfPages(save_dir + file_name)

    if sub_indxs is None:
        sub_indxs = np.arange(subs_n)

    # make sep kers
    sep_s = np.zeros((width_y, width_x, len(sub_indxs)))
    sep_t = np.zeros((nlags, len(sub_indxs)))

    for indx, which_sub in enumerate(sub_indxs):
        k = np.reshape(subs[:, which_sub], (width_y, width_x, tbasis_n))
        k = np.matmul(k, tbasis.T)

        sep_s[..., indx] = np.max(abs(k), axis=2)
        for lag in range(nlags):
            sep_t[lag, indx] = max(k[..., lag].flatten(),
                                        key=abs) / np.max(abs(sep_s))
        bst_lag = np.argmax(abs(sep_t[:, indx]))
        sep_s[..., indx] = k[..., bst_lag]

    # plot ser kers
    # space
    fig_s = make_spatial_plot(sep_s,
                              dims=[len(sub_indxs)],
                              fig_sz=fig_sz_sep_s)
    fig_s.suptitle('Spatial part of %s_layer:%s\nas if they were separable (at best lag)'
                   % (mode, layer), fontsize = 15 + len(sub_indxs) // 10)
    pp.savefig(fig_s, orientation='horizontal')
    plt.close()
    # time
    fig_t = make_temporal_plot(sep_t,
                               dims=[nlags,  len(sub_indxs)],
                               fig_sz=fig_sz_sep_t)
    fig_t.suptitle('Temporal part of %s_layer:%s\nas if they were separable'
                   % (mode, layer), fontsize = 15 + len(sub_indxs) // 10)
    pp.savefig(fig_t, orientation='horizontal')
    plt.close()

    if not only_sep_plot:
        # plot the most general nonsep form
        for indx, which_sub in enumerate(sub_indxs):
            k = np.reshape(subs[:, which_sub], (width_y, width_x, tbasis_n))
            k = np.matmul(k, tbasis.T)

            fig = make_nonsep_plot(k, dims=[nlags, width_x, width_y], fig_sz=fig_sz_nonsep)
            fig.suptitle('%s_layer:%s,  indx_%s.  # %s'
                         % (mode, layer, indx, which_sub), fontsize=30)
            pp.savefig(fig, orientation='horizontal')
            plt.close()
    pp.close()

    print('...plotting done, %s.pdf saved at %s\n' % (file_name, save_dir))


def make_spatial_plot(s, dims, fig_sz):
    [subs_n] = dims
    fig_s = plt.figure(figsize=fig_sz)

    for i in range(subs_n):
        plt.subplot(subs_n // 10 + 1, 10, i + 1)
        k = s[..., i]
        plt.imshow(k, cmap='Greys',
                   vmin=-np.max(abs(k.flatten())),
                   vmax=np.max(abs(k.flatten())))
        plt.axis('off')
    return fig_s



def make_temporal_plot(t, dims, fig_sz):
    [nlags, subs_n] = dims
    fig_t = plt.figure(figsize=fig_sz)
    for i in range(subs_n):
        plt.subplot(subs_n // 10 + 1, 10, i + 1)
        plt.plot(t[:, i], color='b', linewidth=5)
        plt.plot([0, nlags - 1], [0, 0], 'r--')
        plt.xticks([], [])
        plt.yticks([], [])
        if i + 1 > min(subs_n - subs_n % 10, subs_n - 10):
            plt.xticks([0, nlags // 2, nlags],
                       ['-%.0f ms' % (nlags * 1000 / 30),
                        '-%.0f ms' % (nlags // 2 * 1000 / 30), '0'])
    return fig_t


def make_nonsep_plot(k, dims, fig_sz):
    [nlags, x_width, y_width] = dims
    nrows, ncols = 3, max(dims)

    fig = plt.figure(figsize=fig_sz)
    for ll in range(nlags):
        plt.subplot(nrows, ncols, ll + (ncols - nlags) // 2 + 1)
        plt.imshow(k[..., ll].T, cmap="Greys",
                   vmin=-max(abs(k.flatten())), vmax=max(abs(k.flatten())))
        plt.title('(lag = %s)' % ll)
        plt.xticks([], [])
        plt.yticks([], [])
        if ll == 0:
            plt.xlabel('x')
            plt.ylabel('y')

    for xx in range(x_width):
        plt.subplot(nrows, ncols, ncols + xx + abs(ncols - x_width) // 2 + 1)
        plt.imshow(k[xx, ...], cmap="Greys",
                   vmin=-max(abs(k.flatten())), vmax=max(abs(k.flatten())))
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title('(x = %s)' % xx)
        if xx == 0:
            plt.xlabel('time')
            plt.ylabel('y')

    for yy in range(y_width):
        plt.subplot(nrows, ncols, 2*ncols + yy + abs(ncols - x_width) // 2 + 1)
        plt.imshow(k[:, yy, :].T, cmap="Greys",
                   vmin=-max(abs(k.flatten())), vmax=max(abs(k.flatten())))
        plt.title('(y = %s)' % yy)
        plt.xticks([], [])
        plt.yticks([], [])
        if yy == 0:
            plt.xlabel('x')
            plt.ylabel('time')

    return fig


def plot_pred_vs_true(ndn, stim, robs, which_cell,
                      test_indxs, train_indxs, fr_address=[0, 3],
                      rng_width=500, rows_n=2, cols_n=2, save_dir='./plots/'):
    num_pages = nt // (rows_n * cols_n * rng_width) + 1

    out_fr = ndn.generate_prediction(stim, ffnet_n=fr_address[0], layer=fr_address[1])
    [out_ca, tst, trn] = xv_v1(ndn, stim, robs, test_indxs, train_indxs, plot=False)

    pp = PdfPages(save_dir + 'pvt_cell:%s.pdf' % which_cell)

    for page in range(num_pages):
        fig = plt.figure(figsize=(40, 20))
        for ii in range(cols_n * rows_n):
            page_starting_point = page * cols_n * rows_n * rng_width
            end_point = min(nt, page_starting_point + (ii + 1) * rng_width + 1)
            intvl = range(page_starting_point + ii * rng_width, end_point)

            if end_point == nt:
                continue

            plt.subplot(rows_n, cols_n, ii + 1)
            plt.plot(intvl, robs[intvl, which_cell], label='robs', color='g', linewidth=4)
            plt.plot(intvl, out_ca[intvl, which_cell], label='ca', color='b', linestyle='dashdot', linewidth=4)
            plt.plot(intvl, out_fr[intvl, which_cell], label='fr', color='r', linestyle='dashed', linewidth=3)
            plt.ylim(-max(abs(robs[:, which_cell])), max(abs(robs[:, which_cell])))
            plt.title('$r^2$ here: %0.2f %s'
                      % (r_squared(pred=out_ca[intvl, which_cell][:, np.newaxis],
                                   true=robs[intvl, which_cell][:, np.newaxis]) * 100, '%'))
            plt.legend()
        plt.suptitle(
            'Cell # %s     |     ...  $r^2$_tst: %0.2f %s,  $r^2$_trn: %0.2f %s   ...     |     intvl: [%d,  %d]'
            % (which_cell, tst[which_cell], '%', trn[which_cell], '%',
               page_starting_point,
               page_starting_point + (ii + 1) * rng_width), fontsize=40)
        pp.savefig(fig, orientation='horizontal')
        plt.close()
    pp.close()

    return [out_fr, out_ca]


def xv_retina(ndn, stim, robs, data_indxs=None, plot=True):
    if data_indxs is None:
        data_indxs = np.arange(robs.shape[0])

    if robs is list:
        nc = 0
        for tmp_robs in robs:
            nc += tmp_robs.shape[1]
    else:
        nc = robs.shape[1]

    out = ndn.generate_prediction(stim[data_indxs, :])
    r2 = r_squared(true=robs[data_indxs, :], pred=out)

    null_adj_nll = ndn.eval_models(input_data=stim,
                                   output_data=robs,
                                   data_indxs=data_indxs,
                                   nulladjusted=True,
                                   use_gpu=True)
    print('\n\nr2:')
    print('    --> mean: %.4f' % np.mean(r2))
    print('    --> median: %.4f\n' % np.median(r2))

    print('null_adj_NLL:')
    print('    --> mean: %.4f' % np.mean(null_adj_nll))
    print('    --> median: %.4f' % np.median(null_adj_nll))

    if plot:
        plt.figure(figsize=(15, 3))
        plt.subplot(121)
        plt.plot(r2)
        plt.plot([0, nc], [0, 0], 'r--',
                 [0, nc], [1, 1], 'g--')
        plt.xlabel('Neurons')
        plt.ylabel('$R^2$')
        plt.title('Fraction of explained variance (on test indices)')

        plt.subplot(122)
        plt.plot(null_adj_nll)
        plt.plot([0, nc], [0, 0], 'r--',
                 [0, nc], [np.mean(null_adj_nll), np.mean(null_adj_nll)], 'g--')
        plt.xlabel('Neurons')
        plt.xlabel('null adj NLL')
        plt.title('Null adjusted negative log-likelihood (on test indices)')
        plt.show()

    return [r2, null_adj_nll]


def display_layer_info(ndn, pretty_table=True):
    normalization_info = {}
    pos_constraint_info = {}
    partial_fit_info = {}

    for nn in range(len(ndn.network_list)):
        for ll, layer_type in enumerate(ndn.network_list[nn]['layer_types']):
            # get normalization info
            normalization_val = ndn.networks[nn].layers[ll].normalize_weights
            if layer_type in ['sep', 'convsep']:
                if normalization_val == 0:
                    normalization_str = '1st part (filter)'
                elif normalization_val == 1:
                    normalization_str = '2nd part (spatial)'
                elif normalization_val == 2:
                    normalization_str = 'Both (filter + spatial)'
                else:
                    normalization_str = 'No normalization'
            else:
                if normalization_val:
                    normalization_str = 'N'
                else:
                    normalization_str = 'No normalization'

            # get positive constraint info
            pos_constraint_val = ndn.networks[nn].layers[ll].pos_constraint
            if layer_type in ['sep', 'convsep']:
                if pos_constraint_val == 0:
                    pos_constraint_str = '1st part (filter)'
                elif pos_constraint_val == 1:
                    pos_constraint_str = '2nd part (spatial)'
                elif pos_constraint_val == 2:
                    pos_constraint_str = 'Both (filter + spatial)'
                else:
                    pos_constraint_str = 'None'
            else:
                if pos_constraint_val:
                    pos_constraint_str = '+'
                else:
                    pos_constraint_str = 'None'

            if layer_type in ['sep', 'convsep']:
                partial_fit_val = ndn.networks[nn].layers[ll].partial_fit
                if partial_fit_val == 0:
                    partial_fit_str = '1st part (filter)'
                elif partial_fit_val == 1:
                    partial_fit_str = '2nd part (spatial)'
                else:
                    partial_fit_str = 'Everything'
            else:
                partial_fit_str = '---'

            # prepare dicts for printing
            _key = 'net' + str(nn) + 'L' + str(ll) + '_' + layer_type
            normalization_info.update({_key: normalization_str})
            pos_constraint_info.update({_key: pos_constraint_str})
            partial_fit_info.update({_key: partial_fit_str})

    if pretty_table:
        t = PrettyTable(['Layer', 'Normalization', 'Positive Constraint', 'Partial Fit'])
        for lbl, val in sorted(normalization_info.iteritems()):
            t.add_row([lbl, val, pos_constraint_info[lbl], partial_fit_info[lbl]])
        print(t)
    else:
        print("{:<12} {:<30} {:<30} {:<20}\n".format('Layers:',
                                                   'Normalization:',
                                                   'Positive Constraint:',
                                                   'Partial Fit:'))
        for label, val in sorted(normalization_info.iteritems()):
            print("{:<12} {:<30} {:<30} {:<20}".format(label,
                                                       val,
                                                       pos_constraint_info[label],
                                                       partial_fit_info[label]))
    print('\n')


def display_model(ndn):

    nlags, tbasis_n = ndn.networks[0].layers[0].weights.shape

    sker_width = ndn.networks[0].layers[1].filter_dims[1]

    num_conv_kers = ndn.networks[0].layers[1].num_filters
    num_neurons = ndn.networks[-1].layers[-1].num_filters

    num_rows = int(np.ceil(num_conv_kers / 2))
    num_cols = 3

    num_conv_hidden = ndn.network_list[0]['layer_types'][2:].count('conv')

    # Plot the model
    # ____________________________________________________________________________________
    # plotting temporal basis
    tbasis = ndn.networks[0].layers[0].weights
    plt.figure(figsize=(6, 2))
    plt.plot(tbasis)
    plt.plot([0, nlags - 1], [0, 0], 'r--')
    plt.yticks([], [])
    plt.xticks([0, nlags // 2, nlags],
               ['-%.0f ms' % (nlags * 1000 / 30),
                '-%.0f ms' % (nlags // 2 * 1000 / 30), '0'])
    plt.show()

    # plotting conv kernels
    print('_______________________________________________________________________________________________________________')

    if ndn.network_list[0]['layer_types'][1] == 'convsep':
        print('--->    plotting ConvSepLayer:')

        tkers = np.matmul(tbasis, ndn.networks[0].layers[1].weights[:tbasis_n, :])
        skers = ndn.networks[0].layers[1].weights[tbasis_n:, :]

        fig = plt.figure(figsize=(8 * num_cols, 3 * num_rows))
        for i in range(num_rows):
            fig.add_subplot(num_rows, num_cols, (i * num_cols) + 1)
            plt.plot(tkers[:, 2*i], label='# %d' % (2*i))
            if 2*i + 1 < num_conv_kers:
                plt.plot(tkers[:, 2*i + 1], label='# %d' % (2*i + 1))
            plt.title('t_kers')
            plt.legend(loc='best')

            k = np.reshape(skers[:, 2*i], [sker_width, sker_width])
            fig.add_subplot(num_rows, num_cols, (i * num_cols) + 2)
            plt.imshow(k, cmap='Greys',
                       vmin=-max(abs(k.flatten())), vmax=max(abs(k.flatten())))
            plt.colorbar()
            plt.title('s_ker # %d' % (2*i))

            if 2*i + 1 < num_conv_kers:
                k = np.reshape(skers[:, 2*i + 1], [sker_width, sker_width])
                fig.add_subplot(num_rows, num_cols, (i * num_cols) + 3)
                plt.imshow(k, cmap='Greys',
                           vmin=-max(abs(k.flatten())), vmax=max(abs(k.flatten())))
                plt.colorbar()
                plt.title('s_ker # %d' % (2*i + 1))
        plt.show()
    else:
        print('--->    plotting nonsep-ConvLayer (generated at best lag):')

        nonsep_kers = deepcopy(ndn.networks[0].layers[1].weights)

        # make sep kers
        sep_skers = np.zeros((sker_width, sker_width, num_conv_kers))
        sep_tkers = np.zeros((nlags, num_conv_kers))

        for which_ker in range(num_conv_kers):
            k = np.reshape(nonsep_kers[:, which_ker], (sker_width, sker_width, tbasis_n))
            k = np.matmul(k, tbasis.T)

            sep_skers[..., which_ker] = np.max(abs(k), axis=2)
            for lag in range(nlags):
                sep_tkers[lag, which_ker] = max(k[..., lag].flatten(),
                                                key=abs) / np.max(abs(sep_skers))
            bst_lag = np.argmax(abs(sep_tkers[:, which_ker]))
            sep_skers[..., which_ker] = k[..., bst_lag]

        fig = plt.figure(figsize=(9 * num_cols, 3 * num_rows))
        for i in range(num_rows):
            fig.add_subplot(num_rows, num_cols, (i * num_cols) + 1)
            plt.plot(sep_tkers[:, 2*i], label='# %d' % (2*i))
            if 2*i + 1 < num_conv_kers:
                plt.plot(sep_tkers[:, 2*i + 1], label='# %d' % (2*i + 1))
            plt.plot([0, nlags - 1], [0, 0], 'r--')
            plt.yticks([], [])
            plt.xticks([0, nlags // 2, nlags],
                       ['-%.0f ms' % (nlags * 1000 / 30),
                        '-%.0f ms' % (nlags // 2 * 1000 / 30), '0'])
            plt.title('t_kers')
            plt.legend(loc='best')

            k = sep_skers[..., 2*i]
            fig.add_subplot(num_rows, num_cols, (i * num_cols) + 2)
            plt.imshow(k, cmap='Greys',
                       vmin=-max(abs(k.flatten())), vmax=max(abs(k.flatten())))
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title('s_ker # %d' % (2*i))

            if 2*i + 1 < num_conv_kers:
                k = sep_skers[..., 2*i + 1]
                fig.add_subplot(num_rows, num_cols, (i * num_cols) + 3)
                plt.imshow(k, cmap='Greys',
                           vmin=-max(abs(k.flatten())), vmax=max(abs(k.flatten())))
                plt.xticks([], [])
                plt.yticks([], [])
                plt.title('s_ker # %d' % (2*i + 1))
        plt.show()

    # Plotting the rest of the model
    if num_conv_hidden == 0:
        print('\n--->    plotting readout layer:')

        plt.figure(figsize=(16, 2))
        plt.subplot(121)
        plt.imshow(ndn.networks[0].layers[2].weights[:num_conv_kers, :],
                   cmap='Greys', vmin=-1, vmax=1, aspect=num_neurons/num_conv_kers)
        plt.colorbar(aspect='1')

        plt.subplot(122)
        plt.plot(
            np.sum(ndn.networks[0].layers[2].weights[num_conv_kers:, :],
                   axis=1))
        plt.show()

    elif num_conv_hidden == 1:
        print('\n--->    plotting ModLayer, readout layer:')

        mod_n = ndn.networks[0].layers[2].num_filters

        plt.figure(figsize=(18, 3))
        plt.subplot(131)
        k = ndn.networks[0].layers[2].weights
        if (ndn.network_list[0]['pos_constraints'][2] and
                ndn.network_list[0]['normalize_weights'][2]):
            plt.imshow(k, aspect=k.shape[1]/k.shape[0], vmin=0, vmax=1)
        elif (ndn.network_list[0]['pos_constraints'][2] is None and
                ndn.network_list[0]['normalize_weights'][2]):
            plt.imshow(k, aspect=k.shape[1]/k.shape[0], vmin=-1, vmax=1)
        else:
            plt.imshow(k, aspect=k.shape[1]/k.shape[0],
                       vmin=-max(abs(k.flatten())), vmax=max(abs(k.flatten())))
        plt.title('mod layer')
        plt.xlabel('Mods.')
        plt.ylabel('Kernels.')
        plt.colorbar()

        plt.subplot(132)
        k = ndn.networks[0].layers[3].weights[:mod_n, :]
        plt.imshow(k, cmap='Greys', aspect=num_neurons/mod_n,
                   vmin=-max(abs(k.flatten())), vmax=max(abs(k.flatten())))
        plt.title('readout layer: mod part')
        plt.colorbar()
        plt.xlabel('Neurons.')
        plt.ylabel('Mods.')

        plt.subplot(133)
        plt.plot(np.sum(ndn.networks[0].layers[3].weights[mod_n:, :], axis=1))
        plt.title('readout layer: spatial part')
        plt.show()

    elif num_conv_hidden == 2:
        print('\n--->    plotting ModLayers:')

        plt.figure(figsize=(18, 3))

        k = ndn.networks[0].layers[2].weights
        plt.subplot(121)
        if (ndn.network_list[0]['pos_constraints'][2] and
                ndn.network_list[0]['normalize_weights'][2]):
            plt.imshow(k, aspect=k.shape[1]/k.shape[0], vmin=0, vmax=1)
            plt.ylabel('Kernels.')
        elif (ndn.network_list[0]['pos_constraints'][2] is None and
                ndn.network_list[0]['normalize_weights'][2]):
            plt.imshow(k, aspect=k.shape[1]/k.shape[0], vmin=-1, vmax=1)
            plt.ylabel('Kernels.')
        else:
            plt.imshow(k, aspect=k.shape[1]/k.shape[0],
                       vmin=-max(abs(k.flatten())), vmax=max(abs(k.flatten())))
            plt.ylabel('Kernels.')
        plt.title('1st mod layer')
        plt.colorbar()

        k = ndn.networks[0].layers[3].weights
        plt.subplot(122)
        if (ndn.network_list[0]['pos_constraints'][3] and
                ndn.network_list[0]['normalize_weights'][3]):
            plt.imshow(k, aspect=k.shape[1]/k.shape[0], vmin=0, vmax=1)
            plt.xlabel('Mods.')
        elif (ndn.network_list[0]['pos_constraints'][3] is None and
                ndn.network_list[0]['normalize_weights'][3] == 1):
            plt.imshow(k, aspect=k.shape[1]/k.shape[0], vmin=-1, vmax=1)
            plt.xlabel('Mods.')
        else:
            plt.imshow(k, aspect=k.shape[1]/k.shape[0],
                       vmin=-max(abs(k.flatten())), vmax=max(abs(k.flatten())))
            plt.xlabel('Mods.')
        plt.title('2nd mod layer')
        plt.colorbar()
        plt.show()

        print('\n--->    plotting readout layer:')

        mod_n = ndn.networks[0].layers[3].num_filters

        plt.figure(figsize=(14, 2))
        plt.subplot(121)
        plt.imshow(ndn.networks[0].layers[4].weights[:mod_n, :], cmap='Greys',
                   vmin=-1, vmax=1, aspect=num_neurons/mod_n)
        plt.title('readout layer: mod part')
        plt.colorbar()
        plt.xlabel('Neurons')
        plt.ylabel('Mods')

        plt.subplot(122)
        plt.plot(np.sum(ndn.networks[0].layers[4].weights[mod_n:, :], axis=1))
        plt.title('readout layer: spatial part')
        plt.show()
    # ____________________________________________________________________________________

def xv_v1(ndn, stim, robs, test_indxs, train_indxs, plot=True):

    out = ndn.generate_prediction(stim, use_gpu=True)
    r2_tst = r_squared(true=robs, pred=out, data_indxs=test_indxs) * 100
    r2_trn = r_squared(true=robs, pred=out, data_indxs=train_indxs) * 100

    print('\n\ntest r2:')
    print('    --> mean: %.4f %s' % (np.mean(r2_tst), '%'))
    print('    --> median: %.4f %s\n' % (np.median(r2_tst), '%'))

    if plot:
        nc = robs.shape[1]

        plt.figure(figsize=(14, 3.5))
        plt.subplot(121)
        plt.plot(r2_tst)
        plt.plot([0, nc-1], [0, 0], 'r--')
        plt.title('test $r^2$   ...   mean = %0.2f %s' % (np.mean(r2_tst), '%'))


        plt.subplot(122)
        plt.plot(r2_trn)
        plt.plot([0, nc - 1], [0, 0], 'r--')
        plt.title('train $r^2$   ...   mean = %0.2f %s' % (np.mean(r2_trn), '%'))
        plt.show()

    return [out, r2_tst, r2_trn]


def get_ftvr(ndn, weights_to_fit=None, biases_to_fit=None):
    _ftvr = ndn.fit_variables(fit_biases=False)
    num_net = len(ndn.network_list)

    for nn in range(num_net):
        for ll in range(len(ndn.network_list[nn]['layer_sizes'])):
            _ftvr[nn][ll]['weights'] = False

    if weights_to_fit is not None:
        assert len(weights_to_fit) == num_net, 'must be a list with len num_net'

        for nn in range(num_net):
            for ll in weights_to_fit[nn]:
                _ftvr[nn][ll]['weights'] = True

    if biases_to_fit is not None:
        assert len(biases_to_fit) == num_net, 'must be a list with len num_net'

        for nn in range(num_net):
            for ll in biases_to_fit[nn]:
                _ftvr[nn][ll]['biases'] = True

    return _ftvr
