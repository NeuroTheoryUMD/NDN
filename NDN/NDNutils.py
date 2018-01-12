from __future__ import division
import numpy as np
from scipy.linalg import toeplitz


def FFnetwork_params( input_dims = None,
                      layer_sizes = None,
                      ei_layers = None,
                      act_funcs = 'relu',
                      reg_list = None,
                      layers_to_normalize = None,
                      xstim_n = 0,
                      ffnet_n = None,
                      verbose = True,
                      network_type = 'normal',
                      num_conv_layers = 0, # the below are for convolutional network (SIN-NIM)
                      sep_layers = None,
                      conv_filter_widths = None,
                      shift_spacing = 1 ):
    """This generates the information for the network_params dictionary that is passed into
    the constructor for the NetworkNIM. It has the following input arguments:
      -> stim_dims
      -> layer_sizes: list of number of subunits in each layer of network. Last layer should match number of
            neurons (Robs). Each entry can be a 3-d list, if there is a spatio-filter/temporal arrangement.
      -> ei_layers: if this is not none, it should be a list of # of inhibitory units for each layer other than
            the output_layer: so list should be of length one-less than layer_sizes. All the non-inhibitory units
            are of course excitatory, and having 'None' for a layer means it will be unrestricted.
      -> act_funcs: (str or list of strs, optional): activation function for network layers; replicated if a
            single element.
            ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' | 'elu' | 'quad' | 'lin'
      -> xstim_n: data-structure to process (in the case that there are more than one in the model). It should
            be 'None' if the network will be directed internally (see ffnet_n)
      -> ffnet_n: internal network that received input from (has to be None if xstim_n is used)
      This function can also add parameters specific to the SinNIM if num_conv_layers > 0
      -> conv_layers: number of convolutional layers
      -> filter_widths: spatial dimension of filter (if different than stim_dims)
      -> shift_spacing: how much shift in between each convolutional operation
    """

    if layer_sizes is None:
        raise TypeError('Must specify layer_sizes.')

    if xstim_n is not None:
        if not isinstance(xstim_n, list):
            xstim_n = [xstim_n]
    else:
        assert ffnet_n is not None, 'Must assign some input source.'
    if network_type is 'side':
        xstim_n = None

    if ffnet_n is not None:
        if not isinstance(ffnet_n, list):
            ffnet_n = [ffnet_n]

    # Process input_dims, if applicable
    if input_dims is not None:
        input_dims = expand_input_dims_to_3d( input_dims )

    # Build layer_sizes, layer_types, and ei_layers
    num_layers = len(layer_sizes)
    layer_types = ['normal']*num_layers
    for nn in range(num_conv_layers):
        layer_types[nn]='conv'
    if sep_layers is not None:
        if not isinstance(sep_layers, list):
            sep_layers = [sep_layers]
        for nn in sep_layers:
            layer_types[nn]='sep'

    # Establish positivity constraints
    pos_constraints = [False] * num_layers
    num_inh_layers = [0] * num_layers

    # Establish normalization
    norm_weights = [0] * num_layers
    if layers_to_normalize is not None:
        for nn in layers_to_normalize:
            norm_weights[nn] = 1

    if ei_layers is not None:
        for nn in range(len(ei_layers)):
            if ei_layers[nn] >= 0:
                num_inh_layers[nn] = ei_layers[nn]
                if nn < (num_layers-1):
                    pos_constraints[nn+1] = True
    if not isinstance(act_funcs, list):
        act_funcs = [act_funcs] * num_layers

    # Reformat regularization information into regularization for each layer
    reg_initializers = []
    for nn in range(num_layers):
        reg_initializers.append({})
        if reg_list is not None:
            for reg_type, reg_val_list in reg_list.iteritems():
                if not isinstance(reg_val_list, list):
                    if reg_val_list is not None:
                        reg_initializers[nn][reg_type] = reg_val_list
                else:
                    assert len(reg_val_list) == num_layers, 'reg_list length must match number of layers.'
                    if reg_val_list[nn] is not None:
                        reg_initializers[nn][reg_type] = reg_val_list[nn]

    network_params = {
        'network_type': network_type,
        'xstim_n': xstim_n,
        'ffnet_n': ffnet_n,
        'input_dims': input_dims,
        'layer_sizes': layer_sizes,
        'layer_types': layer_types,
        'activation_funcs': act_funcs,
        'pos_constraints': pos_constraints,
        'normalize_weights': norm_weights,
        'num_inh': num_inh_layers,
        'reg_initializers': reg_initializers }

    # if convolutional, add the following SinNIM-specific fields
    if num_conv_layers > 0:
        if not isinstance(conv_filter_widths, list):
            conv_filter_widths = [conv_filter_widths]
        while len(conv_filter_widths) < num_conv_layers:
            conv_filter_widths.append(None)
        network_params['conv_filter_widths'] = conv_filter_widths

        network_params['shift_spacing'] = [shift_spacing]*num_conv_layers

    if verbose:
        if input_dims is not None:
            print( 'Input dimensions: ' + str(input_dims) )
        for nn in range(num_conv_layers):
            s = 'Conv Layer ' + str(nn) + ' (' + act_funcs[nn] + '): [E' + str(layer_sizes[nn]-num_inh_layers[nn])
            s += '/I' + str(num_inh_layers[nn]) + ']'
            if pos_constraints[nn]:
                s += ' +'
            if conv_filter_widths[nn] is not None:
                s += '  \tfilter width = ' + str(conv_filter_widths[nn])
            print(s)
        for nn in range(num_conv_layers,num_layers):
            s = 'Layer ' + str(nn) + ' (' + act_funcs[nn] + '): [E' + str(layer_sizes[nn]-num_inh_layers[nn])
            s += '/I' + str(num_inh_layers[nn]) + ']'
            if pos_constraints[nn]:
                s += ' +'
            print(s)
    return network_params
# END createNIMparams


def expand_input_dims_to_3d(input_size):
    """Utility function to turn inputs into 3-d vectors"""

    if not isinstance(input_size, list):
        input3d = [1, input_size, 1]
    else:
        input3d = input_size
    while len(input3d) < 3:
        input3d.append(1)

    return input3d


def concatenate_input_dims(parent_input_size, added_input_size):
    """Utility function to concatenate two sets of input_dims vectors
    -- parent_input_size can be none, if added_input_size is first
    -- otherwise its assumed parent_input_size is already 3-d, but
        added input size might have to be formatted."""

    cat_dims = expand_input_dims_to_3d( added_input_size )

    if parent_input_size is not None:
        # Sum full vector along the second dimension (first spatial)
        assert parent_input_size[0] == cat_dims[0], 'First dimension of inputs do not agree.'
        assert parent_input_size[2] == cat_dims[2], 'Last dimension of inputs do not agree.'
        cat_dims[1] += parent_input_size[1]

    return cat_dims


def shift_mat_zpad( x, shift, dim=0 ):
    # Takes a vector or matrix and shifts it along dimension dim by amount shift using zero-padding.
    # Positive shifts move the matrix right or down

    assert x.ndim < 3, 'only works in 2 dims or less at the moment.'
    if x.ndim == 1:
        oneDarray = True
        xcopy = np.zeros([len(x), 1])
        xcopy[:, 0] = x
    else:
        xcopy = x.copy()
        oneDarray = False
    sz = list(np.shape(xcopy))

    if sz[0] == 1:
        dim = 1

    if dim == 0:
        if shift >= 0:
            a = np.zeros((shift, sz[1]))
            b = xcopy[0:sz[0]-shift, :]
            xshifted = np.concatenate((a, b), axis=dim)
        else:
            a = np.zeros((-shift, sz[1]))
            b = xcopy[-shift:, :]
            xshifted = np.concatenate((b, a), axis=dim)
    elif dim == 1:
        if shift >= 0:
            a = np.zeros((sz[0], shift))
            b = xcopy[:, 0:sz[1]-shift]
            xshifted = np.concatenate((a, b), axis=dim)
        else:
            a = np.zeros((sz[0], -shift))
            b = xcopy[:, -shift:]
            xshifted = np.concatenate((b, a), axis=dim)

    # If the shift in one direction is bigger than the size of the stimulus in that direction return a zero matrix
    if (dim == 0 and abs(shift) > sz[0]) or (dim == 1 and abs(shift) > sz[1]):
        xshifted = np.zeros(sz)

    # Make into single-dimension if it started that way
    if oneDarray:
        xshifted = xshifted[:,0]

    return xshifted
# END shit_mat_zpad


def create_time_embedding(stim, pdims, up_fac=1, tent_spacing=1):
    """
    # All the arguments starting with a p are part of params structure which I will fix later
    # Takes a Txd stimulus matrix and creates a time-embedded matrix of size Tx(d*L), where L is the desired
    # number of time lags.
    # If stim is a 3d array, the spatial dimensions are folded into the 2nd dimension.
    # Assumes zero-padding.
    # Optional up-sampling of stimulus and tent-basis representation for filter estimation.
    # Note that xmatrix is formatted so that adjacent time lags are adjacent within a time-slice of the xmatrix, thus
    # x(t, 1:nLags) gives all time lags of the first spatial pixel at time t.
    #
    # INPUTS:
    #           stim: simulus matrix (time must be in the first dim).
    #           params: struct of simulus params (see NIM.create_stim_params)
    # OUTPUTS:
    #           xmat: time-embedded stim matrix
    """

    # Note for myself: pdims[0] is nLags and the rest is spatial dimension

    sz = list(np.shape(stim))

    # If there are two spatial dims, fold them into one
    if len(sz) > 2:
        stim = np.reshape(stim, (sz[0], np.prod(sz[1:])))

    # No support for more than two spatial dimensions
    if len(sz) > 3:
        print 'More than two spatial dimensions not supported, but creating xmatrix anyways...'

    # Check that the size of stim matches with the specified stim_params structure
    if np.prod(pdims[1:]) != sz[1]:
        print 'Stimulus dimension mismatch'
        raise ValueError

    modstim = stim.copy()
    # Up-sample stimulus if required
    if up_fac > 1:
        modstim = np.repeat(modstim, up_fac, 0)  # Repeats the stimulus along the time dimension
        sz = list(np.shape(modstim))  # Since we have a new value for time dimension

    # If using tent-basis representation
    if tent_spacing > 1:
        # Create a tent-basis (triangle) filter
        tent_filter = np.append( np.arange(1,tent_spacing)/tent_spacing, 1-np.arange(tent_spacing)/tent_spacing) / tent_spacing
        # Apply to the stimulus
        filtered_stim = np.zeros(sz)
        for ii in range(len(tent_filter)):
            filtered_stim = filtered_stim + shift_mat_zpad(modstim, ii-tent_spacing+1, 0) * tent_filter[ii]
        modstim = filtered_stim

    sz = list(np.shape(modstim))
    lag_spacing = tent_spacing  # If ptent_spacing is not given in input then manually put lag_spacing = 1
    # For temporal-only stimuli (this method can be faster if you're not using tent-basis rep)
    # For myself, add: & tent_spacing is empty (= & isempty...).  Since isempty(tent_spa...) is equivalent to
    # its value being 1 I added this condition to the if below temporarily:
    if sz[1] == 1 and tent_spacing == 1:
        xmat = toeplitz(np.reshape(modstim, (1, sz[0])), np.concatenate((modstim[0], np.zeros(pdims[0] - 1)), axis=0))
    else:  # Otherwise loop over lags and manually shift the stim matrix
        xmat = np.zeros((sz[0], np.prod(pdims)))
        for lag in range(pdims[0]):
            for xx in range(0, sz[1]):
                xmat[:, xx*pdims[0]+lag] = shift_mat_zpad(modstim[:, xx], lag_spacing*lag, 0)

    return xmat
# END create_time_embedding


def spikes_to_Robs( spks, NT, dt ):

    bins_to_use = range(NT+1)*dt
    Robs, bin_edges = np.histogram( spks.flatten(), bins=bins_to_use.flatten() )
    Robs = np.expand_dims( Robs, axis=1 )

    return Robs
