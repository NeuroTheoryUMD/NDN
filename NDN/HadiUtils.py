"""Neural deep network situtation-specific utils by Hadi"""

from __future__ import print_function
from __future__ import division

import os
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import math
from matplotlib.backends.backend_pdf import PdfPages


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

import numpy as np


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

    pp = PdfPages(save_dir + '%s.pdf' % mode)

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

    print('...plotting done, %s.pdf saved at %s\n' % (mode, save_dir))

#    return sep_t, sep_s


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
