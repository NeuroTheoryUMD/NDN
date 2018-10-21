from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt  # plotting
import NDN as NDN


def display_filter3d(ks, filter_dims, options='best_lag'):
    """ks is one dimensional np.array with input_dims. It will display depending on one of many options:
        Single frame options (can be inserted as subplot:
            'best_lag': plot spatial at best lag [default]
            'power': plot power across spatial
        Multi-frame options:

    """

    assert len(filter_dims) == 3, 'input_dims must have 3 dimensions'
    assert ks.shape[0] == np.prod(filter_dims), 'ks is wrong size for input_dims'

    k3d = np.reshape(ks, [filter_dims[2], filter_dims[1], filter_dims[0]])

    if options == 'best_lag':
        best_lag = np.argmax(np.sum(np.sum(np.square(k3d), axis=1), axis=0))
        kspace = k3d[:, :, best_lag]
        Imax = np.max(np.abs(kspace))
        Imin = -Imax
        colormap = 'bwr'
    elif options == 'power':
        kspace = np.sum(np.square(k3d), axis=2)
        Imin, Imax = 0, np.max(kspace)
        colormap = 'Greys'
    else:
        kspace = []
        print('Not valid option.')

    plt.imshow(kspace, cmap=colormap, vmin=Imin, vmax=Imax, interpolation='none')


def sparse_mapping( ndn_mod=None,  layer_target=-1, ffnet_target=None, stim_mag = 100):
    """Note that this will only work with a single input stimulus."""

    assert ndn_mod.num_input_streams == 1, 'This code currently only works with one input stream.'
    if ffnet_target is None:
        ffnet_target = ndn_mod.ffnet_out[0]

    ndims = np.prod(ndn_mod.networks[0].input_dims)
    stim = np.zeros([2*ndims, ndims], dtype='float32')
    for nn in range(ndims):
        stim[nn, nn] = stim_mag
        stim[ndims+nn, nn] = -stim_mag

    preds = ndn_mod.generate_prediction(input_data=stim, ffnet_target=ffnet_target, layer_target=layer_target)
    num_units = preds.shape[1]
    smap = np.reshape(preds, [2]+ndn_mod.networks[0].input_dims+[num_units])

    return smap


def propagate_filter_weights( ndn_mod=None, prop_type='abs', layer_target=-1, ffnet_target=0):
    """Valid propagation types are linear, power, and abs"""

    if ndn_mod is None:
        raise TypeError('Must specify NDN model.')
    ffnet = ndn_mod.networks[ffnet_target]
    nlags = ffnet.input_dims[0]
    num_layers = len(ffnet.layers)
    assert layer_target < num_layers, 'layer_target is too large.'
    if layer_target == -1:
        layer_target = num_layers

    if layer_target == 0:
        if prop_type == 'linear':
            Kprop = ffnet.layers[0].weights
        elif prop_type == 'abs':
            Kprop = np.abs(ffnet.layers[0].weights)
        elif prop_type == 'power':
            Kprop = np.square(ffnet.layers[0].weights)
        else:
            raise TypeError('Invalid prop_type')
    else:
        Kprev = propagate_filter_weights(ndn_mod=ndn_mod, prop_type=prop_type,
                                         layer_target=layer_target-1, ffnet_target=ffnet_target)
        space_dims = ffnet.layers[layer_target-1].filter_dims[1:]
        fdims = ffnet.layers[layer_target].filter_dims[1:]
        layer_type = ndn_mod.network_list[ffnet_target]['layer_types'][layer_target]
        if layer_type == 'normal':
            Kprop = np.zeros(ffnet.input_dims, dtype='float32')

        elif layer_type == 'conv':
            space_dims[0] += fdims[0]-1
            space_dims[1] += fdims[1]-1
            Kprop = np.zeros(space_dims+[nlags], dtype='float32')

    return Kprop
