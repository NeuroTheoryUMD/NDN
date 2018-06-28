"""Utility functions to assist with creating, training and analyzing NDN 
models.
"""

from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.linalg import toeplitz


def ffnetwork_params(
        layer_sizes=None,
        input_dims=None,
        ei_layers=None,
        act_funcs='relu',
        reg_list=None,
        layers_to_normalize=None,
        xstim_n=0,
        ffnet_n=None,
        verbose=True,
        network_type='normal',
        num_conv_layers=0,      # the below are for convolutional network
        num_convsep_layers=0,
        sep_layers=None,
        conv_filter_widths=None,
        shift_spacing=1,
        log_activations=False):
    """generates information for the network_params dict that is passed to the
    NDN constructor.
    
    Args:
        layer_sizes (list of ints): number of subunits in each layer of 
            the network. Last layer should match number of neurons (Robs). Each 
            entry can be a 3-d list, if there is a spatio-filter/temporal 
            arrangement.
        input_dims (list of ints, optional): list of the form 
            [num_lags, num_x_pix, num_y_pix] that describes the input size for 
            the network. If only a single dimension is needed, use 
            [1, input_size, 1].
        ei_layers (`None` or list of ints, optional): if not `None`, it should  
            be a list of the number of inhibitory units for each layer other 
            than the output layer, so list should be of length one less than 
            layer_sizes. All the non-inhibitory units are of course excitatory, 
            and having `None` for a layer means it will be unrestricted.
        act_funcs: (str or list of strs, optional): activation function for 
            network layers; replicated if a single element.
            ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' | 'elu' | 
            'quad' | 'lin'
        reg_list (dict, optional): each key corresponds to a type of 
            regularization (refer to regularization documentation for a 
            complete list). An example using l2 regularization looks like
            {'l2': [l2_layer_0_val, l2_layer_1_val, ..., l2_layer_-1_val}. If
            a single value is given like
            {'l2': l2_val}
            then that value is applied to all layers in the network.           
        layers_to_normalize (list, optional): description
        xstim_n (int or `None`): index into external list of input matrices 
            that specifies which input to process. It should be `None` if the 
            network will be directed internally (see ffnet_n)
        ffnet_n (int or `None`): internal network that this network receives 
            input from (has to be `None` if xstim_n is not `None`)
        verbose (bool, optional): `True` to print network specifications
        network_type (str, optional): specify type of network
            ['normal'] | 'sep'
        num_conv_layers (int, optional): number of convolutional layers
        num_convsep_layers (int, optional): number of convolutional, separable
            layers
        sep_layers (int, optional):
        conv_filter_widths (list of ints, optional): spatial dimension of 
            filter (if different than stim_dims)
        shift_spacing (int, optional): stride used by convolution operation
        log_activations (bool, optional): `True` to log layer activations for
            viewing in tensorboard
        
    Returns:
        dict: params to initialize an `FFNetwork` object
        
    Raises:
        TypeError: If `layer_sizes` is not specified
        TypeError: If both `xstim_n` and `ffnet_n` are `None`
        ValueError: If `reg_list` is a list and its length does not equal 
            the number of layers
        
    """

    if layer_sizes is None:
        raise TypeError('Must specify layer_sizes.')

    if xstim_n is not None:
        if not isinstance(xstim_n, list):
            xstim_n = [xstim_n]
    elif ffnet_n is None:
        TypeError('Must assign some input source.')

    if network_type is 'side':
        xstim_n = None

    if ffnet_n is not None:
        if not isinstance(ffnet_n, list):
            ffnet_n = [ffnet_n]

    # Process input_dims, if applicable
    if input_dims is not None:
        input_dims = expand_input_dims_to_3d(input_dims)

    # Build layer_sizes, layer_types, and ei_layers
    num_layers = len(layer_sizes)
    layer_types = ['normal']*num_layers

    # for now assume all conv layers come after convsep layers
    for nn in range(num_convsep_layers):
        layer_types[nn] = 'convsep'
    for nn in range(num_conv_layers):
        layer_types[num_convsep_layers+nn] = 'conv'

    if sep_layers is not None:
        if not isinstance(sep_layers, list):
            sep_layers = [sep_layers]
        for nn in sep_layers:
            layer_types[nn] = 'sep'

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
            if ei_layers[nn] is not None:
                if ei_layers[nn] >= 0:
                    num_inh_layers[nn] = ei_layers[nn]
                    if nn < (num_layers-1):
                        pos_constraints[nn+1] = True
            else:
                num_inh_layers[nn] = 0
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
                    if len(reg_val_list) != num_layers:
                        ValueError(
                            'reg_list length must match number of layers.')
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
        'normalize_weights': norm_weights,
        'reg_initializers': reg_initializers,
        'num_inh': num_inh_layers,
        'pos_constraints': pos_constraints,
        'log_activations': log_activations}

    # if convolutional, add the following convolution-specific fields
    if num_conv_layers + num_convsep_layers > 0:
        if not isinstance(conv_filter_widths, list):
            conv_filter_widths = [conv_filter_widths]
        while len(conv_filter_widths) < num_conv_layers + num_convsep_layers:
            conv_filter_widths.append(None)
        network_params['conv_filter_widths'] = conv_filter_widths

        network_params['shift_spacing'] = \
            [shift_spacing]*(num_conv_layers + num_convsep_layers)

    if verbose:
        if input_dims is not None:
            print('Input dimensions: ' + str(input_dims))
        for nn in range(num_conv_layers):
            s = 'Conv Layer ' + str(nn) + ' (' + act_funcs[nn] + '): [E' + \
                str(layer_sizes[nn]-num_inh_layers[nn])
            s += '/I' + str(num_inh_layers[nn]) + ']'
            if pos_constraints[nn]:
                s += ' +'
            if conv_filter_widths[nn] is not None:
                s += '  \tfilter width = ' + str(conv_filter_widths[nn])
            print(s)
        for nn in range(num_conv_layers, num_layers):
            s = 'Layer ' + str(nn) + ' (' + act_funcs[nn] + '): [E' + \
                str(layer_sizes[nn]-num_inh_layers[nn])
            s += '/I' + str(num_inh_layers[nn]) + ']'
            if pos_constraints[nn]:
                s += ' +'
            print(s)
    return network_params
# END FFNetwork_params


def expand_input_dims_to_3d(input_size):
    """Utility function to turn inputs into 3-d vectors"""

    if not isinstance(input_size, list):
        input3d = [1, input_size, 1]
    else:
        input3d = input_size[:]
    while len(input3d) < 3:
        input3d.append(1)

    return input3d


def concatenate_input_dims(parent_input_size, added_input_size):
    """Utility function to concatenate two sets of input_dims vectors
    -- parent_input_size can be none, if added_input_size is first
    -- otherwise its assumed parent_input_size is already 3-d, but
    added input size might have to be formatted.
    
    Args:
        parent_input_size (type): description
        added_input_size (type): description
        
    Returns:
        type: description
        
    Raises:
    
    """

    cat_dims = expand_input_dims_to_3d(added_input_size)

    if parent_input_size is not None:
        # Sum full vector along the second dimension (first spatial)
        assert parent_input_size[0] == cat_dims[0], \
            'First dimension of inputs do not agree.'
        assert parent_input_size[2] == cat_dims[2], \
            'Last dimension of inputs do not agree.'
        cat_dims[1] += parent_input_size[1]

    return cat_dims


def shift_mat_zpad(x, shift, dim=0):
    """Takes a vector or matrix and shifts it along dimension dim by amount 
    shift using zero-padding. Positive shifts move the matrix right or down.
    
    Args:
        x (type): description
        shift (type): description
        dim (type): description
        
    Returns:
        type: description
            
    Raises:
            
    """

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

    # If the shift in one direction is bigger than the size of the stimulus in
    # that direction return a zero matrix
    if (dim == 0 and abs(shift) > sz[0]) or (dim == 1 and abs(shift) > sz[1]):
        xshifted = np.zeros(sz)

    # Make into single-dimension if it started that way
    if oneDarray:
        xshifted = xshifted[:,0]

    return xshifted
# END shift_mat_zpad


def create_time_embedding(stim, pdims, up_fac=1, tent_spacing=1):
    """All the arguments starting with a p are part of params structure which I 
    will fix later.
    
    Takes a Txd stimulus matrix and creates a time-embedded matrix of size 
    Tx(d*L), where L is the desired number of time lags. If stim is a 3d array, 
    the spatial dimensions are folded into the 2nd dimension. 
    
    Assumes zero-padding.
     
    Optional up-sampling of stimulus and tent-basis representation for filter 
    estimation.
    
    Note that xmatrix is formatted so that adjacent time lags are adjacent 
    within a time-slice of the xmatrix, thus x(t, 1:nLags) gives all time lags 
    of the first spatial pixel at time t.
    
    Args:
        stim (type): simulus matrix (time must be in the first dim).
        pdims (list/array): length(3) list of stimulus dimensions
        up_fac (type): description
        tent_spacing (type): description
        
    Returns:
        numpy array: time-embedded stim matrix
        
    """

    # Note for myself: pdims[0] is nLags and the rest is spatial dimension

    sz = list(np.shape(stim))

    # If there are two spatial dims, fold them into one
    if len(sz) > 2:
        stim = np.reshape(stim, (sz[0], np.prod(sz[1:])))

    # No support for more than two spatial dimensions
    if len(sz) > 3:
        print('More than two spatial dimensions not supported, but creating' +
              'xmatrix anyways...')

    # Check that the size of stim matches with the specified stim_params
    # structure
    if np.prod(pdims[1:]) != sz[1]:
        print('Stimulus dimension mismatch')
        raise ValueError

    modstim = stim.copy()
    # Up-sample stimulus if required
    if up_fac > 1:
        # Repeats the stimulus along the time dimension
        modstim = np.repeat(modstim, up_fac, 0)
        # Since we have a new value for time dimension
        sz = list(np.shape(modstim))

    # If using tent-basis representation
    if tent_spacing > 1:
        # Create a tent-basis (triangle) filter
        tent_filter = np.append(
            np.arange(1, tent_spacing) / tent_spacing,
            1-np.arange(tent_spacing)/tent_spacing) / tent_spacing
        # Apply to the stimulus
        filtered_stim = np.zeros(sz)
        for ii in range(len(tent_filter)):
            filtered_stim = filtered_stim + \
                            shift_mat_zpad(modstim,
                                           ii-tent_spacing+1,
                                           0) * tent_filter[ii]
        modstim = filtered_stim

    sz = list(np.shape(modstim))
    lag_spacing = tent_spacing

    # If tent_spacing is not given in input then manually put lag_spacing = 1
    # For temporal-only stimuli (this method can be faster if you're not using
    # tent-basis rep)
    # For myself, add: & tent_spacing is empty (= & isempty...).
    # Since isempty(tent_spa...) is equivalent to its value being 1 I added
    # this condition to the if below temporarily:
    if sz[1] == 1 and tent_spacing == 1:
        xmat = toeplitz(np.reshape(modstim, (1, sz[0])),
                        np.concatenate((modstim[0], np.zeros(pdims[0] - 1)),
                                       axis=0))
    else:  # Otherwise loop over lags and manually shift the stim matrix
        xmat = np.zeros((sz[0], np.prod(pdims)))
        for lag in range(pdims[0]):
            for xx in range(0, sz[1]):
                xmat[:, xx*pdims[0]+lag] = shift_mat_zpad(
                    modstim[:, xx], lag_spacing * lag, 0)

    return xmat
# END create_time_embedding


def generate_spike_history(robs, nlags, neg_constraint=True, reg_par=0,
                           xstim_n=1):
    """Will generate X-matrix that contains Robs information for each cell. It will
    use the default resolution of robs, and simply go back a certain number of lags.
    To have it applied to the corresponding neuron, will need to use add_layers"""

    NC = robs.shape[1]

    Xspk = create_time_embedding(shift_mat_zpad(robs, 1, dim=0), pdims=[nlags, NC, 1])
    ffnetpar = ffnetwork_params(layer_sizes=[NC], input_dims=[nlags, NC, 1],
                                act_funcs='lin', reg_list={'d2t': reg_par},
                                xstim_n=xstim_n, verbose=False,
                                network_type='spike_history')
    # Constrain spike history terms to be negative
    ffnetpar['pos_constraints'] = [neg_constraint]
    ffnetpar['num_inh'] = [NC]
    ffnetpar['layer_types'] = ['spike_history']

    return Xspk, ffnetpar
# END generate_spike_history


def generate_xv_folds(nt, fold=5, num_blocks=3):
    """Will generate unique and cross-validation indices, but subsample in each block
        NT = number of time steps
        fold = fraction of data (1/fold) to set aside for cross-validation
        num_blocks = how many blocks to sample fold validation from"""

    test_inds = []
    start_ind = 0
    NTblock = np.floor(nt/num_blocks)
    # Pick middle of block for XV
    tstart = np.floor(NTblock * (np.floor(fold / 2) / fold))
    XVlength = np.round(NTblock / fold)
    for bl in range(num_blocks):
        test_inds = test_inds + range(int(start_ind+tstart), int(start_ind+tstart+XVlength))
        start_ind = start_ind + NTblock

    test_inds = np.array(test_inds)
    train_inds = np.array(list(set(range(0, nt)) - set(test_inds)))  # Better way to setdiff?

    return train_inds, test_inds


def spikes_to_robs(spks, num_time_pts, dt):
    """
    Description
    
    Args:
        spks (type): description
        num_time_pts (type): description
        dt (type): description
        
    Returns:
        type: description

    """

    bins_to_use = range(num_time_pts + 1) * dt
    robs, bin_edges = np.histogram(spks.flatten(), bins=bins_to_use.flatten())
    robs = np.expand_dims(robs, axis=1)

    return robs


# GPU picking

import subprocess, re, os, sys

def run_command(cmd):
    """Run command, return output as string."""

    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")


def list_available_gpus():
    """Returns list of available GPU ids."""

    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse " + line
        result.append(int(m.group("gpu_id")))
    return result


def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result


def pick_gpu_lowest_memory():
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu


def setup_one_gpu():
    assert not 'tensorflow' in sys.modules, "GPU setup must happen before importing TensorFlow"
    gpu_id = pick_gpu_lowest_memory()
    print("   ...picking GPU # " + str(gpu_id))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def setup_no_gpu():
    if 'tensorflow' in sys.modules:
        print("Warning, GPU setup must happen before importing TensorFlow")
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

def assign_gpu():
    print('*******************************************************************************************')

    print('---> getting list of available GPUs:')
    print(list_available_gpus())
    print('\n---> getting GPU memory map:')
    print(gpu_memory_map())
    print('\n---> setting up GPU with largest available memory:')
    setup_one_gpu()

    print('*******************************************************************************************')

    print('\nDone!')
    return pick_gpu_lowest_memory()
