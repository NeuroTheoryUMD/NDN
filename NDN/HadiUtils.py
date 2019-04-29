# coding=utf-8
"""Neural deep network situtation-specific utils by Hadi"""

from __future__ import print_function
from __future__ import division

import os
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import math
from matplotlib.backends.backend_pdf import PdfPages
from prettytable import PrettyTable
from matplotlib.animation import FuncAnimation
import seaborn as sns
import datetime


def convert_time(time_in_secs):

    d = time_in_secs // 86400
    h = (time_in_secs - d * 86400) // 3600
    m = (time_in_secs - d * 86400 - h * 3600) // 60
    s = time_in_secs - d * 86400 - h * 3600 - m * 60

    print("\nd / hh:mm:ss   --->   %d / %d:%d:%d\n" % (d, h, m, s))


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
        data_indxs = np.arange(true.shape[0])

    ss_res = np.sum(np.square(true[data_indxs, :] - pred[data_indxs, :]), axis=0) / len(data_indxs)
    ss_tot = np.var(true[data_indxs, :], axis=0)

    _nonzero_inds = np.where(ss_tot != 0)[0]
    _zero_inds = np.where(ss_tot == 0)[0]

    r2 = np.zeros(true.shape[1])

    r2[_nonzero_inds] = 1 - np.divide(ss_res[_nonzero_inds], ss_tot[_nonzero_inds])
    r2[_zero_inds] = 1 - ss_res[_zero_inds]

    return r2


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

    nx, ny = dims
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

    file_name = datetime.datetime.now().strftime("%Y:%m:%d_%H:%M")+'_%s.pdf' % mode
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


def plot_pred_vs_true(ndn, stim, robs, which_cell, test_indxs, train_indxs,
                      fr_address, rng_width=1000, rows_n=2, cols_n=2,
                      save_dir='./plots/', file_name=None,
                      style='darkgrid', facecolor='white'):

    _allowed_styles = ['white', 'whitegrid', 'dark', 'darkgrid']

    if style not in _allowed_styles:
        raise valueError('invalid style ''%s''' % style)

   # if style in ['white', 'whitegrid', 'dark', 'darkgrid']:
  #      jtplot.reset()
      #  sns.set_style(style)
      #  sns.set(rc={'figure.facecolor': facecolor})
  #  else:
  #      jtplot.style(theme=style, grid=False, ticks=True, figsize=(6, 4))

    if not os.path.isdir(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))

    nt = robs.shape[0]
    num_pages = nt // (rows_n * cols_n * rng_width) + 1

    out_fr = ndn.generate_prediction(stim, ffnet_target=fr_address[0], layer_target=fr_address[1])
    [out_ca, tst, trn] = xv_v1(ndn, stim, robs, test_indxs, train_indxs, plot=False)

    if file_name is None:
        file_name = 'pvt_cell:%s.pdf' % which_cell
    pp = PdfPages(save_dir + file_name)

    # find the upper and lower bounds for ylim in plots
    _robs_ub = max(robs[np.concatenate((train_indxs, test_indxs)) , which_cell])
    _robs_lb = min(robs[np.concatenate((train_indxs, test_indxs)), which_cell])

    _out_fr_ub = max(out_fr[np.concatenate((train_indxs, test_indxs)) , which_cell])
    _out_fr_lb = min(out_fr[np.concatenate((train_indxs, test_indxs)), which_cell])

    _ylim_ub = max(_robs_ub, _out_fr_ub)
    _ylim_lb = min(_robs_lb, _out_fr_lb)

    for page in range(num_pages):
        fig = plt.figure(figsize=(30, rows_n * 10 // cols_n))
        for ii in range(cols_n * rows_n):
            page_starting_point = page * cols_n * rows_n * rng_width
            end_point = min(nt, page_starting_point + (ii + 1) * rng_width + 1)
            intvl = range(page_starting_point + ii * rng_width, end_point)

            if end_point == nt:
                continue

            if style in ['white', 'whitegrid', 'dark', 'darkgrid']:
                plt.style.use('seaborn-' + style)
                plt.style.context('poster')
                plt.subplot(rows_n, cols_n, ii + 1)
                plt.plot(intvl, robs[intvl, which_cell], label='cobs', color='darkslategrey', linewidth=1.5)
                plt.plot(intvl, out_ca[intvl, which_cell], label='ca', color='r', linewidth=3)
                plt.plot(intvl, out_fr[intvl, which_cell], label='fr', color='royalblue', linewidth=1)
            else:
                plt.subplot(rows_n, cols_n, ii + 1)
                plt.plot(intvl, robs[intvl, which_cell], label='robs', color='g', linewidth=3)
                plt.plot(intvl, out_ca[intvl, which_cell], label='ca', color='b', linewidth=5)
                plt.plot(intvl, out_fr[intvl, which_cell], label='fr', color='y', linestyle='dashed', linewidth=2)
            plt.ylim(_ylim_lb, _ylim_ub)
            plt.title('$r^2$ here: %0.2f %s'
                      % (r_squared(pred=out_ca[intvl, which_cell][:, np.newaxis],
                                   true=robs[intvl, which_cell][:, np.newaxis]) * 100, '%'))
            plt.legend()
        plt.suptitle(
            'Cell # %s     |     ...  $r^2$_tst: %0.2f %s,  $r^2$_trn: %0.2f %s   ...     |     intvl: [%d,  %d]'
            % (which_cell, tst[which_cell], '%', trn[which_cell], '%',
               page_starting_point,
               page_starting_point + (ii + 1) * rng_width), fontsize=30)
        plt.draw()
        pp.savefig(fig, orientation='horizontal', facecolor=facecolor)
        plt.close()
    pp.close()

    # go back to onedork
   # if style in ['white', 'whitegrid', 'dark', 'darkgrid']:
   #     jtplot.style(theme='onedork', grid=False, ticks=True, figsize=(6, 4))

    return [out_fr, out_ca]



def plot_pred_vs_true_simplified(
    datas, lbls, lws, colors, which_cell,
    rng_width=1000, rows_n=2, cols_n=2,
    save_dir='./plots/', file_name=None,
    style='darkgrid', facecolor='white'):

    _allowed_styles = ['white', 'whitegrid', 'dark', 'darkgrid']

    if style not in _allowed_styles:
        raise valueError('invalid style ''%s''' % style)

    assert (len(datas) == len(lbls) == len(lws) == len(colors)), 'must have same length'

    if not os.path.isdir(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))

    nt = datas[0].shape[0]
    num_pages = int(np.floor(nt // (rows_n * cols_n * rng_width)))

    if file_name is None:
        file_name = 'pvt_cell:%d.pdf' % which_cell
    pp = PdfPages(save_dir + file_name)

    # find the upper and lower bounds for ylim in plots
    _ylim_ub = 0
    _ylim_lb = 0
    for data in datas:
        _ylim_ub = max(_ylim_ub, max(data[: , which_cell]))
        _ylim_lb = min(_ylim_lb, min(data[: , which_cell]))

    for page in range(num_pages):
        fig = plt.figure(figsize=(40, rows_n * 10 // cols_n))
        for ii in range(cols_n * rows_n):
            page_starting_point = page * cols_n * rows_n * rng_width
            end_point = min(nt, page_starting_point + (ii + 1) * rng_width + 1)
            intvl = range(page_starting_point + ii * rng_width, end_point)

            if end_point == nt:
                continue

            plt.style.use('seaborn-' + style)
            plt.style.context('poster')
            plt.subplot(rows_n, cols_n, ii + 1)
            for indx, data in enumerate(datas):
                plt.plot(intvl, data[intvl, which_cell], label=lbls[indx],
                         color=colors[indx], linewidth=lws[indx])
            plt.ylim(_ylim_lb, _ylim_ub)
            plt.title('$r^2$ here: %0.2f %s'
                      % (r_squared(pred=datas[1][intvl, which_cell][:, np.newaxis],
                                   true=datas[0][intvl, which_cell][:, np.newaxis]) * 100, '%'))
            plt.legend()

        plt.suptitle(
            '|     cell # %d     ...     |     intvl: [%d,  %d]     ...     |'
            % (which_cell, page_starting_point,
               page_starting_point + (ii + 1) * rng_width), fontsize=30)
        plt.draw()
        pp.savefig(fig, orientation='horizontal', facecolor=facecolor)
        plt.close()
    pp.close()



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
        plt.ylabel('null adj NLL')
        plt.title('Null adjusted negative log-likelihood (on test indices)')
        plt.show()

    return [out, r2, null_adj_nll]


def display_layer_info(ndn, pretty_table=True):
    architecture = 'FF'

    for nn in range(len(ndn.network_list)):
        if ndn.network_list[nn]['network_type'] in ['side', 'temporal_side']:
            architecture = 'SCAFF'

    print('Architecture = %s,  batch size = %d,  time spread = %d \n'
          % (architecture, ndn.batch_size, ndn.time_spread))

    normalization_info = {}
    pos_constraint_info = {}
    partial_fit_info = {}
    act_func_info = {}
    ei_info = {}
    te_info = {}
    conv_info = {}

    for nn in range(len(ndn.network_list)):
        for ll, layer_type in enumerate(ndn.network_list[nn]['layer_types']):
            # get normalization info
            normalization_val = ndn.networks[nn].layers[ll].normalize_weights
            if layer_type in ['sep', 'convsep', 'gabor']:
                if normalization_val == 0:
                    normalization_str = '1st part (F)'
                elif normalization_val == 1:
                    normalization_str = '2nd part (S)'
                elif normalization_val == 2:
                    normalization_str = 'Both (F + S)'
                else:
                    normalization_str = 'None'
            else:
                if normalization_val:
                    normalization_str = 'Yes'
                else:
                    normalization_str = 'None'

            # get positive constraint info
            pos_constraint_val = ndn.networks[nn].layers[ll].pos_constraint
            if layer_type in ['sep', 'convsep', 'gabor']:
                if pos_constraint_val == 0:
                    pos_constraint_str = '1st part (F)'
                elif pos_constraint_val == 1:
                    pos_constraint_str = '2nd part (S)'
                elif pos_constraint_val == 2:
                    pos_constraint_str = 'Both (F + S)'
                else:
                    pos_constraint_str = 'None'
            else:
                if pos_constraint_val:
                    pos_constraint_str = '+'
                else:
                    pos_constraint_str = 'None'

            if layer_type in ['sep', 'convsep', 'gabor']:
                partial_fit_val = ndn.networks[nn].layers[ll].partial_fit
                if partial_fit_val == 0:
                    partial_fit_str = '1st part (F)'
                elif partial_fit_val == 1:
                    partial_fit_str = '2nd part (S)'
                else:
                    partial_fit_str = 'Everything'
            else:
                partial_fit_str = '---'

            # get act_func info
            act_func_str = ndn.network_list[nn]['activation_funcs'][ll]
            if act_func_str == 'leaky_relu':
                act_func_str = act_func_str + ' (alpha = %s)' % ndn.networks[nn].layers[ll].nl_param

            # get E/I info
            num_inh = ndn.network_list[nn]['num_inh'][ll]
            num_exc = ndn.network_list[nn]['layer_sizes'][ll] - num_inh
            ei_str = '[E' + str(num_exc) + '/I' + str(num_inh) + '] -> ' + str(num_inh + num_exc)

            # get time expand info
            if 'time_expand' in ndn.network_list[nn].keys():
                te_str = str(ndn.network_list[nn]['time_expand'][ll])
            else:
                te_str = '---'  ### change this to None once you made temporal side network

            # get conv info
            if layer_type in ['temporal', 'ca_tent']:
                conv_str = ('- / ' + str(ndn.network_list[nn]['dilation'][ll])
                            + ' / ' + str(ndn.network_list[nn]['ca_tent_widths'][ll]))
            elif layer_type in ['conv', 'convsep', 'gabor', 'biconv']:
                conv_str = (str(ndn.network_list[nn]['stride'][ll])
                            + ' / ' + str(ndn.network_list[nn]['dilation'][ll])
                            + ' / ' + str(ndn.network_list[nn]['conv_filter_widths'][ll]))
            else:
                conv_str = '---'

            # prepare dicts for printing
            _key = str(nn) + str(ll) + '_' + layer_type
            normalization_info.update({_key: normalization_str})
            pos_constraint_info.update({_key: pos_constraint_str})
            partial_fit_info.update({_key: partial_fit_str})
            act_func_info.update({_key: act_func_str})
            ei_info.update({_key: ei_str})
            te_info.update({_key: te_str})
            conv_info.update({_key: conv_str})

    if pretty_table:
        t = PrettyTable(['Layer', 'Normalization',
                         'Pos Cnstrnt', 'Partial Fit',
                         'Act Func', 'E/I',
                         'Time Expand', 'S / D / W'])
        for lbl, val in sorted(normalization_info.iteritems()):
            t.add_row([lbl, val,
                       pos_constraint_info[lbl], partial_fit_info[lbl],
                       act_func_info[lbl], ei_info[lbl],
                       te_info[lbl], conv_info[lbl]])
        print(t)
    else:
        print("{:<12} {:<30} {:<30} {:<20}\n".format('Layers:',
                                                     'Normalization:',
                                                     'Positive Constraint:',
                                                     'Partial Fit:',
                                                     'Act Func:',
                                                     'E/I',
                                                     'Time Expand', 'S / D / W'))
        for label, val in sorted(normalization_info.iteritems()):
            print("{:<12} {:<30} {:<30} {:<20}".format(label,
                                                       val,
                                                       pos_constraint_info[label],
                                                       partial_fit_info[label],
                                                       act_func_info[label],
                                                       ei_info[label],
                                                       te_info[label],
                                                       conv_info[label]))
    print('\n')


def display_model(ndn, mod_struct='scaff'):
    """
    :param ndn: obv.
    :param mod_struct: allowed_modes = ['scaff', 'ff']
    :return:
    """

    if type(mod_struct) == str:
        if mod_struct == 'scaff':
            address_dict = {'tbasis': [0, 0], 'stkers': [1, 0], 'readout': [-1, -1]}
        elif mod_struct == 'ff':
            address_dict = {'tbasis': [0, 0], 'stkers': [0, 1], 'readout': [-1, -1]}
        else:
            raise ValueError('not supported model structure')
    elif type(mod_struct) == dict:
        address_dict = mod_struct
    else:
        raise ValueError('mod_struct should be either a dict of addresses or a str')

    tbasis_address = address_dict['tbasis']
    stkers_address = address_dict['stkers']
    readout_address = address_dict['readout']

    nlags, tbasis_n = ndn.networks[tbasis_address[0]].layers[tbasis_address[1]].weights.shape

    sker_width = ndn.networks[stkers_address[0]].layers[stkers_address[1]].filter_dims[1]

    num_conv_kers = ndn.networks[stkers_address[0]].layers[stkers_address[1]].num_filters
    num_neurons = ndn.networks[readout_address[0]].layers[readout_address[1]].num_filters

    num_rows = int(np.ceil(num_conv_kers / 2))
    num_cols = 3

    num_conv_hidden = ndn.network_list[stkers_address[0]]['layer_types'][stkers_address[1]+1:].count('conv')

    # Plot the model
    # ____________________________________________________________________________________
    # plotting temporal basis
    tbasis = ndn.networks[tbasis_address[0]].layers[tbasis_address[1]].weights
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

    if ndn.network_list[stkers_address[0]]['layer_types'][stkers_address[1]] == 'convsep':
        print('--->    plotting ConvSepLayer:')

        tkers = np.matmul(tbasis, ndn.networks[stkers_address[0]].layers[stkers_address[1]].weights[:tbasis_n, :])
        skers = ndn.networks[stkers_address[0]].layers[stkers_address[1]].weights[tbasis_n:, :]

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

        nonsep_kers = deepcopy(ndn.networks[stkers_address[0]].layers[stkers_address[1]].weights)

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

            # take care of ON vs OFF (all spatial > 0)
            if max(sep_skers[..., which_ker].flatten(), key=abs) < 0:
                sep_skers[..., which_ker] *= -1
                sep_tkers[:, which_ker] *= -1

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
        plt.imshow(ndn.networks[readout_address[0]].layers[readout_address[1]].weights[:num_conv_kers, :],
                   cmap='Greys', vmin=-1, vmax=1, aspect=num_neurons/num_conv_kers)
        plt.colorbar(aspect='1')

        plt.subplot(122)
        plt.plot(
            np.sum(ndn.networks[readout_address[0]].layers[readout_address[1]].weights[num_conv_kers:, :],
                   axis=1))
        plt.show()

    elif num_conv_hidden == 1:
        print('\n--->    plotting ModLayer, readout layer:')

        mod_n = ndn.networks[stkers_address[0]].layers[stkers_address[1]+1].num_filters

        plt.figure(figsize=(18, 3))
        plt.subplot(131)
        k = ndn.networks[stkers_address[0]].layers[stkers_address[1]+1].weights
        if (ndn.network_list[stkers_address[0]]['pos_constraints'][stkers_address[1]+1] and
                ndn.network_list[stkers_address[0]]['normalize_weights'][stkers_address[1]+1]):
            plt.imshow(k, aspect=k.shape[1]/k.shape[0], vmin=0, vmax=1)
        elif (ndn.network_list[stkers_address[0]]['pos_constraints'][stkers_address[1]+1] is None and
                ndn.network_list[stkers_address[0]]['normalize_weights'][stkers_address[1]+1]):
            plt.imshow(k, aspect=k.shape[1]/k.shape[0], vmin=-1, vmax=1)
        else:
            plt.imshow(k, aspect=k.shape[1]/k.shape[0],
                       vmin=-max(abs(k.flatten())), vmax=max(abs(k.flatten())))
        plt.title('mod layer')
        plt.xlabel('Mods.')
        plt.ylabel('Kernels.')
        plt.colorbar()

        plt.subplot(132)
        k = ndn.networks[readout_address[0]].layers[readout_address[1]].weights[:mod_n, :]
        plt.imshow(k, cmap='Greys', aspect=num_neurons/mod_n,
                   vmin=-max(abs(k.flatten())), vmax=max(abs(k.flatten())))
        plt.title('readout layer: mod part')
        plt.colorbar()
        plt.xlabel('Neurons.')
        plt.ylabel('Mods.')

        plt.subplot(133)
        plt.plot(np.sum(ndn.networks[readout_address[0]].layers[readout_address[1]].weights[mod_n:, :], axis=1))
        plt.title('readout layer: spatial part')
        plt.show()

    elif num_conv_hidden == 2:
        print('\n--->    plotting ModLayers:')

        plt.figure(figsize=(18, 3))

        k = ndn.networks[stkers_address[0]].layers[stkers_address[1]+1].weights
        plt.subplot(121)
        if (ndn.network_list[stkers_address[0]]['pos_constraints'][stkers_address[1]+1] and
                ndn.network_list[stkers_address[0]]['normalize_weights'][stkers_address[1]+1]):
            plt.imshow(k, aspect=k.shape[1]/k.shape[0], vmin=0, vmax=1)
            plt.ylabel('Kernels.')
        elif (ndn.network_list[stkers_address[0]]['pos_constraints'][stkers_address[1]+1] is None and
                ndn.network_list[stkers_address[0]]['normalize_weights'][stkers_address[1]+1]):
            plt.imshow(k, aspect=k.shape[1]/k.shape[0], vmin=-1, vmax=1)
            plt.ylabel('Kernels.')
        else:
            plt.imshow(k, aspect=k.shape[1]/k.shape[0],
                       vmin=-max(abs(k.flatten())), vmax=max(abs(k.flatten())))
            plt.ylabel('Kernels.')
        plt.title('1st mod layer')
        plt.colorbar()

        k = ndn.networks[stkers_address[0]].layers[stkers_address[1]+2].weights
        plt.subplot(122)
        if (ndn.network_list[stkers_address[0]]['pos_constraints'][stkers_address[1]+2] and
                ndn.network_list[stkers_address[0]]['normalize_weights'][stkers_address[1]+2]):
            plt.imshow(k, aspect=k.shape[1]/k.shape[0], vmin=0, vmax=1)
            plt.xlabel('Mods.')
        elif (ndn.network_list[stkers_address[0]]['pos_constraints'][stkers_address[1]+2] is None and
                ndn.network_list[stkers_address[0]]['normalize_weights'][stkers_address[1]+2] == 1):
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

        mod_n = ndn.networks[stkers_address[0]].layers[-1].num_filters

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


def make_gif(data_to_plot, frames=None, interval=120, dt=25, fig_sz=(8, 6), dpi=100, sns_style=None,
             cmap='Greys', mode='stim', scale=None, file_name=None, save_dir='./gifs/', row_n=None, col_n=None):
    if sns_style is not None:
        sns.set_style(sns_style)

    if frames is None:
        if mode in ['stim', 'stim_multi', 'velocity_field']:
            frames = np.arange(data_to_plot.shape[0])
        elif mode == 'subs':
            frames = np.arange(data_to_plot.shape[-2])

    if not os.path.isdir(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))

    if file_name is None:
        file_name = mode + '.gif'

    if mode == 'stim':
        # start the fig
        fig, _ax = plt.subplots()
        fig.set_size_inches(fig_sz)
        # fig.set_tight_layout(True)

        # find absmax for vmin/vmax and plot the first frame
        _abs_max = np.max(abs(data_to_plot))
        _plt = _ax.imshow(data_to_plot[0, ...],
                          cmap=cmap, vmin=-_abs_max, vmax=_abs_max)
        fig.colorbar(_plt)

        # start anim object
        anim = FuncAnimation(fig, _gif_update, fargs=[_plt, _ax, data_to_plot, mode],
                             frames=frames, interval=interval)
        anim.save(save_dir + file_name, dpi=dpi, writer='imagemagick')

    elif mode == 'stim_multi':
        if row_n is None:
            raise ValueError('For multiple stims must enter row_n')

        num_stim = data_to_plot.shape[-1]
        if col_n is None:
            col_n = int(np.ceil(num_stim / row_n))

        _plt_dict = {}

        fig, axes = plt.subplots(row_n, col_n)
        if fig_sz is None:
            fig.set_size_inches((col_n * 2, row_n * 2))
        else:
            fig.set_size_inches((fig_sz[0], fig_sz[1]))

        _abs_max = np.max(abs(data_to_plot))
        for ii in range(row_n):
            for jj in range(col_n):
                which_stim = ii * col_n + jj
                if which_stim >= num_stim:
                    continue

                axes[ii, jj].xaxis.set_ticks([])
                axes[ii, jj].yaxis.set_ticks([])
                tmp_plt = axes[ii, jj].imshow(data_to_plot[0, ..., which_stim], cmap=cmap,
                                              vmin=-_abs_max, vmax=_abs_max)
                _plt_dict.update({'ax_%s_%s' % (ii, jj): tmp_plt})

        # start anim object
        anim = FuncAnimation(fig, _gif_update, fargs=[fig, _plt_dict, data_to_plot, dt, row_n, col_n, mode],
                             frames=frames, interval=interval)
        anim.save(save_dir + file_name, dpi=dpi, writer='imagemagick')

    elif mode == 'velocity_field':
        k = data_to_plot.copy()
        k /= k.std()

        num_arr = k.shape[-1]
        grd = k.shape[1]

        if col_n is None:
            col_n = int(np.ceil(num_arr / row_n))

        _plt_dict = {}

        fig, axes = plt.subplots(row_n, col_n)
        if fig_sz is None:
            fig.set_size_inches((col_n * 2, row_n * 2))
        else:
            fig.set_size_inches((fig_sz[0], fig_sz[1]))

        xx, yy = np.mgrid[0:grd, 0:grd]

        for ii in range(row_n):
            for jj in range(col_n):
                which_arr = ii * col_n + jj
                if which_arr >= num_arr:
                    continue

                # estimate scale
                tmp = -1
                max_lag = -1
                for lag in range(k.shape[0]):
                    tmp_cc = np.sqrt(np.square(k[lag, ..., 0, which_arr])
                                     + np.square(k[lag, ..., 1, which_arr]))
                    if np.max(tmp_cc) > tmp:
                        tmp = np.max(tmp_cc)
                        max_lag = lag
                cc = np.sqrt(np.square(k[max_lag, ..., 0, which_arr])
                             + np.square(k[max_lag, ..., 1, which_arr]))

                # axes indexing
                if row_n > 1 and col_n > 1:
                    ax = axes[ii, jj]
                elif row_n > 1 and col_n == 1:
                    ax = axes[ii]
                elif row_n == 1 and col_n > 1:
                    ax = axes[jj]
                elif row_n == 1 and col_n == 1:
                    ax = axes
                else:
                    raise ValueError('row_n, col_n should be integers greater than or equal to 1')

                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])

                uu, vv = k[-1, ..., 0, which_arr], k[-1, ..., 1, which_arr]
                if scale is None:
                    scale = 8 * np.max(cc)
                else:
                    scale = scale[ii, jj]
                    cc = np.sqrt(np.square(uu) + np.square(vv))

                tmp_plt = ax.quiver(xx, yy, uu, vv, cc, alpha=1, scale=scale, cmap='PuBu')
                ax.scatter(xx, yy, s=0.01)

                ax.set_xlim(-1, grd)
                ax.set_ylim(-1, grd)

                ax.set_aspect('equal')

                ax.set_title('# %d' % which_arr)
                _plt_dict.update({'ax_%s_%s' % (ii, jj): tmp_plt})

        # start anim object
        anim = FuncAnimation(fig, _gif_update,
                             fargs=[fig, _plt_dict, k, dt, scale, row_n, col_n, mode],
                             frames=frames, interval=interval)
        anim.save(save_dir + file_name, dpi=dpi, writer='imagemagick')

    elif mode == 'subs':
        width_y, width_x, nlags, ker_n = data_to_plot.shape
        col_n = int(np.ceil(np.sqrt(ker_n)))
        row_n = int(np.ceil(ker_n / col_n))

        _plt_dict = {}

        fig, axes = plt.subplots(row_n, col_n)
        if fig_sz is None:
            fig.set_size_inches((col_n * 2, row_n * 2))
        else:
            fig.set_size_inches((fig_sz[0], fig_sz[1]))

        _abs_max = np.max(abs(data_to_plot))
        for ii in range(row_n):
            for jj in range(col_n):
                which_sub = ii * col_n + jj
                if which_sub >= ker_n:
                    continue

                axes[ii, jj].xaxis.set_ticks([])
                axes[ii, jj].yaxis.set_ticks([])
                tmp_plt = axes[ii, jj].imshow(data_to_plot[..., 0, which_sub], cmap=cmap,
                                              vmin=-_abs_max, vmax=_abs_max)
                _plt_dict.update({'ax_%s_%s' % (ii, jj): tmp_plt})

        # start anim object
        anim = FuncAnimation(fig, _gif_update, fargs=[fig, _plt_dict, data_to_plot, dt, None, None, mode],
                             frames=frames, interval=interval)
        anim.save(save_dir + file_name, dpi=dpi, writer='imagemagick')

    else:
        raise ValueError, 'Not implemented yet.'

    plt.close()
    print('...your GIF is done! "%s" was saved at %s.' % (file_name, save_dir))


def _gif_update(tt, _fig_or_plt_like, _ax_like, data_to_plot, dt, scale=None, row_n=None, col_n=None, mode='stim'):
    nlags = data_to_plot.shape[0]
    time_remaining = np.rint((nlags - tt) * dt)
    lbl = '- {0} ms'.format(time_remaining)

    if mode == 'stim':
        _plt, _ax = _fig_or_plt_like, _ax_like

        _plt.set_data(data_to_plot[tt, :])
        _ax.set_xlabel(lbl)

        return _plt, _ax

    elif mode == 'stim_multi':
        if row_n is None:
            raise ValueError('For multiple stims must enter row_n')
        num_stim = data_to_plot.shape[-1]
        if col_n is None:
            col_n = int(np.ceil(num_stim / row_n))

        _fig, _plt_dict = _fig_or_plt_like, _ax_like

        for ii in range(row_n):
            for jj in range(col_n):
                which_stim = ii * col_n + jj
                if which_stim >= num_stim:
                    continue
                kk = data_to_plot[tt, ..., which_stim]
                _plt_dict['ax_%s_%s' % (ii, jj)].set_data(kk)
        _fig.suptitle(lbl, fontsize=50)

        return _plt_dict, _fig

    elif mode == 'velocity_field':
        if row_n is None:
            raise ValueError('For velocity fields must enter row_n')
        num_arr = data_to_plot.shape[-1]
        if col_n is None:
            col_n = int(np.ceil(num_arr / row_n))

        _fig, _plt_dict = _fig_or_plt_like, _ax_like

        for ii in range(row_n):
            for jj in range(col_n):
                which_arr = ii * col_n + jj
                if which_arr >= num_arr:
                    continue

                current_uu = data_to_plot[tt, ..., 0, which_arr]
                current_vv = data_to_plot[tt, ..., 1, which_arr]
                if scale is not None:
                    current_cc = np.sqrt(np.square(current_uu) + np.square(current_vv))
                    _plt_dict['ax_%s_%s' % (ii, jj)].set_UVC(current_uu, current_vv, current_cc)
                else:
                    _plt_dict['ax_%s_%s' % (ii, jj)].set_UVC(current_uu, current_vv)

        _fig.suptitle(lbl, fontsize=30)

        return _plt_dict, _fig

    elif mode == 'subs':
        ker_n = data_to_plot.shape[3]

        col_n = int(np.ceil(np.sqrt(ker_n)))
        row_n = int(np.ceil(ker_n / col_n))

        _fig, _plt_dict = _fig_or_plt_like, _ax_like

        for ii in range(row_n):
            for jj in range(col_n):
                which_sub = ii * col_n + jj
                if which_sub >= ker_n:
                    continue
                k = data_to_plot[..., tt, which_sub]
                _plt_dict['ax_%s_%s' % (ii, jj)].set_data(k)
        _fig.suptitle(lbl, fontsize=50)

        return _plt_dict, _fig
    else:
        raise ValueError('Not implemented yet.')


def make_synth_stim(lambdas, thetas, omega, frames_n, width):
    gabors_n = len(lambdas) * len(thetas)
    params = np.zeros((2, gabors_n))

    # make params
    for ii in range(gabors_n):
        ss = ii // len(thetas)
        oo = ii % len(thetas)

        params[0, ii] = lambdas[ss]
        params[1, ii] = thetas[oo]

    # make gabors
    ctr = width // 2
    _pi = np.math.pi

    rng = np.arange(width ** 2)

    yy = rng // width - ctr
    xx = rng % width - ctr

    xx_prime = (np.matmul(yy[:, np.newaxis], np.sin(params[1, :][np.newaxis, :]))
                + np.matmul(xx[:, np.newaxis], np.cos(params[1, :][np.newaxis, :])))

    omega_rad = np.radians(omega)
    synth_stim = np.zeros((frames_n, width, width, gabors_n))

    # start gabors and evolve in time
    for tt in range(1, frames_n):
        tmp_gabors = np.sin(2 * _pi * xx_prime / params[0, :] + omega_rad * tt)
        # TODO: normalization fix later
       # tmp_gabors -= np.mean(tmp_gabors, axis=0)
       # tmp_gabors /= np.std(tmp_gabors[:, 0], axis=0)
        synth_stim[tt, ...] = np.reshape(tmp_gabors, (width, width, -1))

    return synth_stim


def get_st_subs(ndn):
    tbasis = ndn.networks[0].layers[0].weights
    nlags, tbasis_n = tbasis.shape
    stkers = ndn.networks[1].layers[0].weights
    [_, width_x, width_y] = ndn.networks[1].layers[0].filter_dims
    ker_n = stkers.shape[1]

    st_subs = np.zeros((width_y, width_x, nlags, ker_n))

    for which_sub in range(ker_n):
        k = np.reshape(stkers[:, which_sub], (width_y, width_x, tbasis_n))
        st_subs[..., which_sub] = np.matmul(k, tbasis.T)

    return st_subs


def propagte_weights(ndn, mod_struct='scaff', num_conv_mod_layers=None):
    """
    :param ndn: Obv.
    :param mod_struct: model structure
    :param num_conv_mod_layers: Number of mod layers
    :return: dict containing propagated weights throughout network

    """

    if mod_struct == 'scaff':
        address_dict = {'tbasis': [0, 0], 'stkers': [1, 0], 'readout': [-1, -1]}
    elif mod_struct == 'ff':
        address_dict = {'tbasis': [0, 0], 'stkers': [0, 1], 'readout': [-1, -1]}
    else:
        raise ValueError('allowed model structures: SCAFF, FF')

   # if address_dict is None:
   #     address_dict = {'tbasis': [0, 0], 'stkers': [0, 1], 'readout': [0, -1]}

    tbasis_address = address_dict['tbasis']
    tbasis = ndn.networks[tbasis_address[0]].layers[tbasis_address[1]].weights
    nlags, tbasis_n = tbasis.shape

    # kers
    stkers_address = address_dict['stkers']
    stkers = ndn.networks[stkers_address[0]].layers[stkers_address[1]].weights
    [_, width, _] = ndn.networks[stkers_address[0]].layers[stkers_address[1]].filter_dims
    ker_n = stkers.shape[1]

    if num_conv_mod_layers is None:
        num_conv_mod_layers = ndn.network_list[stkers_address[0]]['layer_types'][stkers_address[1]+1:].count('conv')

    st_subs = np.zeros((width, width, nlags, ker_n))

    for which_sub in range(ker_n):
        k = np.reshape(stkers[:, which_sub], (width, width, tbasis_n))
        st_subs[..., which_sub] = np.matmul(k, tbasis.T)

    out_dict = {}
    out_dict.update({'net%dL%d_subs' % (stkers_address[0], stkers_address[1]): st_subs})


    # mods
    for mm in range(1, num_conv_mod_layers + 1):
        mod_width = ndn.networks[stkers_address[0]].layers[stkers_address[1] + mm].filter_dims[1]
        dilation = ndn.networks[stkers_address[0]].layers[stkers_address[1] + mm].dilation
        mod_n = ndn.networks[stkers_address[0]].layers[stkers_address[1] + mm].num_filters
        mod_n_before = ndn.networks[stkers_address[0]].layers[stkers_address[1] + mm - 1].num_filters

        pixels_info = np.zeros((mod_width, mod_width, mod_n_before, mod_n))

        for ii in range(mod_n_before * mod_width ** 2):
            alpha_y, alpha_x, alpha_filt = np.unravel_index(ii, (mod_width, mod_width, mod_n_before))
            pixels_info[alpha_y, alpha_x, alpha_filt,
            :] = ndn.networks[stkers_address[0]].layers[stkers_address[1] + mm].weights[ii, :]

        _key = sorted(out_dict.keys())[-1]
        _last_st = out_dict[_key]
        _ei_mask = ndn.networks[stkers_address[0]].layers[stkers_address[1] + mm - 1].ei_mask
        _last_st_ei = _last_st * _ei_mask
        _last_width = _last_st.shape[0]

        new_width = _last_width + 2 * (mod_width // 2) * dilation
        st_mods = np.zeros((new_width, new_width, nlags, mod_n))

        for which_mod in range(mod_n):
            _tmp_embd = np.zeros((new_width ** 2, nlags))
            for indx_x, alpha_x in enumerate(range(0, dilation * mod_width, dilation)):
                for indx_y, alpha_y in enumerate(range(0, dilation * mod_width, dilation)):
                    x_rng = [alpha_x, alpha_x + _last_width]
                    y_rng = [alpha_y, alpha_y + _last_width]
                    k = np.matmul(_last_st_ei, pixels_info[indx_y, indx_x, :, which_mod])
                    k = np.reshape(k, (-1, nlags))
                    _tmp_embd += space_embedding(k, [new_width, new_width], x_rng, y_rng)
            _tmp_embd = np.reshape(_tmp_embd, (new_width, new_width, nlags))
            st_mods[..., which_mod] = _tmp_embd

        out_dict.update({'net%dL%d_mods' % (stkers_address[0], stkers_address[1] + mm): st_mods})


    # cells
    if 'readout' in address_dict:
        _key = sorted(out_dict.keys())[-1]
        _last_st = out_dict[_key]
        _last_width = _last_st.shape[0]

        readout_address = address_dict['readout']
        mod_n = ndn.networks[readout_address[0]].layers[readout_address[1] - 1].num_filters
        st_cells = np.matmul(_last_st, ndn.networks[readout_address[0]].layers[readout_address[1]].weights[:mod_n, :])

        out_dict.update({'net%dL%d_cells' % (readout_address[0], readout_address[1]): st_cells})

        nc = ndn.networks[readout_address[0]].layers[readout_address[1]].weights.shape[1]
        nx, ny = ndn.input_sizes[0][1:]

        # cells embedded
        st_cells_embd = np.zeros((ny, nx, nlags, nc))
        ros = ndn.networks[readout_address[0]].layers[readout_address[1]].weights[mod_n:, :]

        for mu in range(nc):
            _tmp_embd = np.zeros((ny*nx, nlags))
            for ii in range(ny*nx):
                pos_y, pos_x = np.unravel_index(ii, (ny, nx))
                _tmp_embd += ros[ii, mu] * space_embedding(
                    np.reshape(st_cells[..., mu], (-1, nlags)),
                    dims=[nx, ny],
                    x_rng=[pos_x - _last_width // 2, pos_x + _last_width // 2 + 1],
                    y_rng=[pos_y - _last_width // 2, pos_y + _last_width // 2 + 1])

            st_cells_embd[..., mu] = np.reshape(_tmp_embd, (ny, nx, nlags))

        out_dict.update({'net%dL%d_cells_embd' % (readout_address[0], readout_address[1]): st_cells_embd})

    return out_dict


def get_gabor(params, width, plot=True, gabor_per_plot=None):
    """
    :param params: gabor params
    :param width:
    :param plot:
    :param gabor_per_plot:
    :return:
    """

    _lambda = params[0, :]
    _theta = params[1, :]
    _phi = params[2, :]
    _sigma = params[3, :]

    if params.shape[0] < 5:
        _gamma = np.ones((1, params.shape[1]))
    else:
        _gamma = params[-1, :]

    _pi = np.pi
    ctr = width // 2
    rng = np.arange(width**2)

    yy = rng // width - ctr
    xx = rng % width - ctr

    xx_prime = (np.matmul(yy[:, np.newaxis], np.sin(_theta[np.newaxis, :]))
                + np.matmul(xx[:, np.newaxis], np.cos(_theta[np.newaxis, :])))
    yy_prime = (np.matmul(yy[:, np.newaxis], np.cos(_theta[np.newaxis, :]))
                - np.matmul(xx[:, np.newaxis], np.sin(_theta[np.newaxis, :])))

    exp = np.exp(-(xx_prime ** 2 + (_gamma * yy_prime) ** 2) / (2 * _sigma ** 2))
    gabor = exp * np.sin(2 * _pi * xx_prime / _lambda + _phi)

    if plot:
        if gabor_per_plot is None:
            gabor_per_plot = params.shape[1]

        num_plots = int(np.ceil(params.shape[1] / gabor_per_plot))
        plt_sz = int(np.ceil(np.sqrt(gabor_per_plot)))

        for pp in range(num_plots):
            plt.figure(figsize=(plt_sz, plt_sz))
            for ii in range(gabor_per_plot):
                plt.subplot(plt_sz, plt_sz, ii + 1)
                which_gabor = ii + pp*gabor_per_plot
                k = np.reshape(gabor[:, which_gabor], [width, width])
                plt.imshow(k, cmap='Greys', vmin=-max(abs(k.flatten())), vmax=max(abs(k.flatten())))
                plt.axis('off')
            plt.suptitle('plt # %s,   FROM  %s  to  %s' % (pp, pp*gabor_per_plot, (pp+1)*gabor_per_plot), fontsize=20)
            plt.show()
    return gabor.astype('float32')


def get_ei_indxs(ndn, scaff_net_address=1):
    levels_n = len(ndn.network_list[scaff_net_address]['layer_sizes'])

    inh_ind = []
    for ii in range(levels_n):
        _level_sz = ndn.network_list[scaff_net_address]['layer_sizes'][ii]
        _num_inh = ndn.network_list[scaff_net_address]['num_inh'][ii]
        _num_exc = _level_sz - _num_inh
        if ii == 0:
            rng = range(_num_exc,
                        _num_exc + _num_inh)
        else:
            rng = range(int(inh_ind[-1]) + 1 + _num_exc,
                        int(inh_ind[-1]) + 1 + _num_exc + _num_inh)
        inh_ind = np.concatenate((inh_ind, rng), axis=0)
    inh_ind = inh_ind.astype(int)

    nf_tot = sum(ndn.network_list[scaff_net_address]['layer_sizes'])
    exc_ind = np.delete(np.arange(nf_tot), inh_ind)

    return [exc_ind, inh_ind]


def get_ei_depth(k, level_sizes, inh_sizes, mode='norm'):
    lvls_n = len(level_sizes)
    ei_mass = np.zeros((2, lvls_n))
    measured_ei_depth = np.zeros(2)

    for indx, _level_width in enumerate(level_sizes):
        _num_inh = inh_sizes[indx]
        _num_exc = _level_width - _num_inh
        if indx == 0:
            inh_rng = range(_num_exc, _num_exc + _num_inh)
            exc_rng = range(0, _num_exc)
            all_rng = range(0, _level_width)
        else:
            _accumulated = sum(level_sizes[:indx])
            inh_rng = range(_accumulated + _num_exc,
                            _accumulated + _num_exc + _num_inh)
            exc_rng = range(_accumulated,
                            _accumulated + _num_exc)
            all_rng = range(_accumulated,
                            _accumulated + _level_width)
        # 0 for exc, 1 for inh
        if mode == 'norm':
            if np.linalg.norm(k[all_rng]) > 0:
                ei_mass[0, indx] = np.linalg.norm(k[exc_rng]) / np.linalg.norm(k[all_rng])
                ei_mass[1, indx] = np.linalg.norm(k[inh_rng]) / np.linalg.norm(k[all_rng])
            else:
                ei_mass[0, indx] = 0
                ei_mass[1, indx] = 0
        elif mode == 'sum':
            if np.sum(k[all_rng]) > 0:
                ei_mass[0, indx] = np.sum(k[exc_rng]) / np.sum(k[all_rng])
                ei_mass[1, indx] = np.sum(k[inh_rng]) / np.sum(k[all_rng])
            else:
                ei_mass[0, indx] = 0
                ei_mass[1, indx] = 0

    if np.sum(ei_mass[0, :]) > 0:
        measured_ei_depth[0] = (np.sum(ei_mass[0, :] * np.arange(1, lvls_n + 1))
                                / np.sum(ei_mass[0, :])) - 1
    else:
        measured_ei_depth[0] = 0
    if np.sum(ei_mass[1, :]) > 0:
        measured_ei_depth[1] = (np.sum(ei_mass[1, :] * np.arange(1, lvls_n + 1))
                                / np.sum(ei_mass[1, :])) - 1
    else:
        measured_ei_depth[1] = 0

    return [ei_mass, measured_ei_depth]


def make_ei_depth_plot(ndn, readout_address=None, readout_type='hadi_readout', save_fig=False,
                       save_dir='./plots/', file_name='el_level_depth.png'):
    if readout_address is None:
        readout_address = [2, 0]

    level_sizes = ndn.network_list[readout_address[0]-1]['layer_sizes']
    num_inh = ndn.network_list[readout_address[0]-1]['num_inh']

    levels_n = len(level_sizes)
    nf_tot = sum(level_sizes)

    if readout_type == 'hadi_readout':
        rof = deepcopy(ndn.networks[readout_address[0]].layers[readout_address[1]].weights)
    elif readout_type == 'sep':
        rof = deepcopy(ndn.networks[readout_address[0]].layers[readout_address[1]].weights[:nf_tot, :])
    else:
        raise TypeError('Not implemented yet')

    nc = rof.shape[1]

    measured_ei_depth = np.zeros((2, nc))
    measured_ei_mass = np.zeros((2, levels_n, nc))
    for ii in range(nc):
        measured_ei_mass[..., ii], measured_ei_depth[:, ii] = get_ei_depth(rof[:, ii], level_sizes, num_inh)

    # make the plot
    sns.set_style('darkgrid')

    fig = plt.figure(figsize=(16, 8))
    exc_plt = plt.hist(measured_ei_depth[0, :], bins=25, label='EXC depth', color='r')
    inh_plt = plt.hist(measured_ei_depth[1, :], bins=50, label='INH depth', fc=(0, 0.2, 0.8, 0.7))

    y1 = max(max(exc_plt[0]), max(inh_plt[0]))

    plt.plot([np.mean(measured_ei_depth[0, :]), np.mean(measured_ei_depth[0, :])], [0, y1],
             color='y', linewidth=4, linestyle='dashdot', label='mean (EXC)')
    plt.plot([np.mean(measured_ei_depth[1, :]), np.mean(measured_ei_depth[1, :])], [0, y1],
             color='c', linewidth=4, linestyle='dashdot', label='mean (INH)')

    plt.xlabel('depth', fontsize=25)
    plt.ylabel('count', fontsize=25)
    # plt.xticks([], [])
    plt.yticks([], [])
    plt.xlim(0, levels_n - 1)
    plt.legend()
    plt.title('E/I level depth', fontsize=25)
    if save_fig:
        fig.savefig(save_dir + file_name, facecolor='grey')
    plt.show()

    return [measured_ei_mass, measured_ei_depth]


def sep_to_nonsep(ndn, mode='to_ndn'):

    # kers
    nlags, tbasis_n = ndn.networks[0].layers[0].weights.shape
    coeffs = ndn.networks[1].layers[0].weights[:tbasis_n, :]
    t_kers = np.matmul(ndn.networks[0].layers[0].weights, coeffs)
    s_kers = ndn.networks[1].layers[0].weights[tbasis_n:, :]

    width = ndn.networks[1].layers[0].filter_dims[1]
    ker_n = ndn.networks[1].layers[0].num_filters

    if mode == 'to_plot':
        st_kers = np.zeros((width ** 2, nlags, ker_n), dtype='float32')

        for which_sub in range(ker_n):
            for lag in range(nlags):
                st_kers[..., lag, which_sub] = s_kers[:, which_sub] * t_kers[lag, which_sub]
    elif mode == 'to_ndn':
        st_kers = np.zeros((width ** 2, tbasis_n, ker_n), dtype='float32')

        for which_sub in range(ker_n):
            for bb in range(tbasis_n):
                st_kers[..., bb, which_sub] = s_kers[:, which_sub] * coeffs[bb, which_sub]
    else:
        raise ValueError('wrong mode (should be either to_plot or to_ndn)')

    return st_kers.reshape(-1, ker_n)


def save_mod(ndn, data_dir, name, xv):
    time_str = datetime.datetime.now().strftime("%Y:%m:%d_%H:%M")

    if ndn.noise_dist == 'gaussian':
        file_name = time_str + name + '_(xv:%.4f%s)' % (float(np.mean(xv)), '%')
    elif ndn.noise_dist == 'poisson':
        file_name = time_str + name + '_(xv:%.4f)' % np.mean(xv)
    else:
        raise ValueError('not implemented yet')
    ndn.save_model(data_dir + 'pkld_mods/' + file_name)


def xv_save(ndn, inputs, outputs, tst_ind, trn_ind, data_dir, save_name):
    noise_dist = ndn.noise_dist

    if noise_dist == 'gaussian':
        [out, tst_xv, trn_xv] = xv_v1(ndn, inputs, outputs, tst_ind, trn_ind, plot=True)
    elif noise_dist == 'poisson':
        [out, tst_xv] = xv_retina(ndn, inputs, outputs, tst_ind, plot=True)
        # TODO: add trn_xv
        trn_xv = None
    else:
        raise ValueError('not implemented yet')

    save_mod(ndn, data_dir, save_name, tst_xv)

    return [out, tst_xv, trn_xv]