from __future__ import division

import numpy as np
import scipy.sparse as sp


def create_tikhonov_matrix(stim_dims, reg_type, boundary_conditions=None):
    """
    Usage: Tmat = create_Tikhonov_matrix(stim_dims, reg_type, boundary_cond)

    Creates a matrix specifying a an L2-regularization operator of the form
    ||T*k||^2, where T is a matrix and k is a vector of parameters. Currently 
    only supports second derivative/Laplacian operations

    Args:
        stim_dims (list of ints): dimensions associated with the target 
            stimulus, in the form [num_lags, num_x_pix, num_y_pix]
        reg_type (str): specify form of the regularization matrix
            'd2xt' | 'd2x' | 'd2t'
        boundary_conditions (None): would ideally be a dictionary with each reg 
            type listed; currently unused

    Returns:
        scipy array: matrix specifying the desired Tikhonov operator

    Notes:
        The method of computing sparse differencing matrices used here is 
        adapted from Bryan C. Smith's and Andrew V. Knyazev's function 
        "laplacian", available here: 
        http://www.mathworks.com/matlabcentral/fileexchange/27279-laplacian-in-1d-2d-or-3d
        Written in Matlab by James McFarland, adapted into python by Dan Butts
        
    """

    # first dimension is assumed to represent time lags
    nLags = stim_dims[0]

    # additional dimensions are spatial (Nx and Ny)
    nPix = stim_dims[1] * stim_dims[2]
    allowed_reg_types = ['d2xt', 'd2x', 'd2t']

    # assert (ischar(reg_type) && ismember(reg_type, allowed_reg_types),
    # 'not an allowed regularization type');

    # has_split = ~isempty(stim_params.split_pts);
    et = np.ones([1, nLags], dtype=np.float32)
    ex = np.ones([1, stim_dims[1]], dtype=np.float32)
    ey = np.ones([1, stim_dims[2]], dtype=np.float32)

    # Boundary conditions (currently not implemented)
    # if isinf(stim_params.boundary_conds(1)) # if temporal dim has free boundary
    # et[0, [0, -1]] = 0  # constrain temporal boundary to zero: all else are free
    # if isinf(stim_params.boundary_conds(2)) # if first spatial dim has free boundary
    ex[0, [0, -1]] = 0
    # if isinf(stim_params.boundary_conds(3)); # if second spatial dim has free boundary
    ey[0, [0, -1]] = 0

    if nPix == 1:  # for 0-spatial-dimensional stimuli can only do temporal

        assert reg_type is 'd2t', 'Can only do temporal reg for stimuli without spatial dims'

        Tmat = sp.spdiags(np.concatenate((et, -2 * et, et), axis=0), [-1, 0, 1], nLags, nLags)
        # if stim_params.boundary_conds(1) == -1 # if periodic boundary cond
        #    Tmat(end, 1) = 1;
        #    Tmat(1, end) = 1;

    elif stim_dims[2] == 1:  # for 1 - spatial dimensional stimuli
        if reg_type is 'd2t':
            assert nLags > 1, 'No d2t regularization possible with no lags.'

            # D1t = spdiags([et - 2 * et et], [-1 0 1], nLags, nLags)';
            D1t = sp.coo_matrix.transpose(sp.coo_matrix(
                sp.spdiags(np.concatenate((et, -2 * et, et), axis=0), [-1, 0, 1], nLags, nLags)))
            #    if stim_params.boundary_conds(1) == -1 % if periodic boundary cond
            #        D1t(end, 1) = 1;
            #        D1t(1, end) = 1;
            Ix = sp.eye(stim_dims[1])
            Tmat = sp.kron(Ix, D1t)

        elif reg_type is 'd2x':
            It = sp.eye(nLags)
            # Matlab code: D1x = spdiags([ex -2*ex ex], [-1 0 1], nPix(1), nPix(1))';
            D1x = sp.coo_matrix.transpose(sp.coo_matrix(
                sp.spdiags(np.concatenate((ex, -2 * ex, ex), axis=0), [-1, 0, 1], stim_dims[1], stim_dims[1])))
            # if stim_params.boundary_conds(2) == -1 % if periodic boundary cond
            #    D1x(end, 1) = 1;
            #    D1x(1, end) = 1;

            Tmat = sp.kron(D1x, It)

        elif reg_type is 'd2xt':
            # D1t = spdiags([et - 2 * et et], [-1 0 1], nLags, nLags)';
            D1t = sp.coo_matrix.transpose(sp.coo_matrix(
                sp.spdiags(np.concatenate((et, -2 * et, et), axis=0), [-1, 0, 1], nLags, nLags)))
            # if stim_params.boundary_conds(1) == -1 % if periodic boundary cond
            #    D1t(end, 1) = 1;
            #    D1t(1, end) = 1;

            # D1x = spdiags([ex - 2 * ex ex], [-1 0 1], nPix(1), nPix(1))';
            D1x = sp.coo_matrix.transpose(sp.coo_matrix(
                sp.spdiags(np.concatenate((ex, -2 * ex, ex), axis=0), [-1, 0, 1], stim_dims[1], stim_dims[1])))
            # if stim_params.boundary_conds(2) == -1 % if periodic boundary cond
            #    D1x(end, 1) = 1;
            #    D1x(1, end) = 1;

            It = sp.eye(nLags)
            Ix = sp.eye(stim_dims[1])
            Tmat = sp.kron(Ix, D1t) + sp.kron(D1x, It)

    else:  # for stimuli with 2-spatial dimensions
        if reg_type is 'd2t':
            assert nLags > 1, 'No d2t regularization possible with no lags.'
            # D1t = spdiags([et - 2 * et et], [-1 0 1], nLags, nLags)';
            D1t = sp.coo_matrix.transpose(sp.coo_matrix(
                sp.spdiags(np.concatenate((et, -2 * et, et), axis=0), [-1, 0, 1], nLags, nLags)))
            # if stim_params.boundary_conds(1) == -1 % if periodic boundary cond
            #    D1t(end, 1) = 1;
            #    D1t(1, end) = 1;

            Ix = sp.eye(stim_dims[1])
            Iy = sp.eye(stim_dims[2])
            Tmat = sp.kron(Iy, sp.kron(Ix, D1t))

        elif reg_type is 'd2x':
            It = sp.eye(nLags)
            Ix = sp.eye(stim_dims[1])
            Iy = sp.eye(stim_dims[2])
            # D1x = spdiags([ex - 2 * ex ex], [-1 0 1], nPix(1), nPix(1))';
            D1x = sp.coo_matrix.transpose(sp.coo_matrix(
                sp.spdiags(np.concatenate((ex, -2 * ex, ex), axis=0), [-1, 0, 1], stim_dims[1], stim_dims[1])))
            # if stim_params.boundary_conds(2) == -1 % if periodic boundary cond
            #    D1x(end, 1) = 1
            #    D1x(1, end) = 1

            # D1y = spdiags([ey - 2 * ey ey], [-1 0 1], nPix(2), nPix(2))';
            D1y = sp.coo_matrix.transpose(sp.coo_matrix(
                sp.spdiags(np.concatenate((ey, -2 * ey, ey), axis=0), [-1, 0, 1], stim_dims[2], stim_dims[2])))
            # if stim_params.boundary_conds(3) == -1 % if periodic boundary cond
            #    D1y(end, 1) = 1;
            #    D1y(1, end) = 1;

            Tmat = sp.kron(Iy, sp.kron(D1x, It)) + sp.kron(D1y, sp.kron(Ix, It))

        elif reg_type is 'd2xt':
            It = sp.eye(nLags)
            Ix = sp.eye(stim_dims[1])
            Iy = sp.eye(stim_dims[2])
            # D1t = spdiags([et - 2 * et et], [-1 0 1], nLags, nLags)';
            D1t = sp.coo_matrix.transpose(sp.coo_matrix(
                sp.spdiags(np.concatenate((et, -2 * et, et), axis=0), [-1, 0, 1], nLags, nLags)))
            # if stim_params.boundary_conds(1) == -1 # if periodic boundary cond
            #    D1t(end, 1) = 1
            #    D1t(1, end) = 1

            # D1x = spdiags([ex - 2 * ex ex], [-1 0 1], nPix(1), nPix(1))';
            D1x = sp.coo_matrix.transpose(sp.coo_matrix(
                sp.spdiags(np.concatenate((ex, -2 * ex, ex), axis=0), [-1, 0, 1], stim_dims[1], stim_dims[1])))
            # if stim_params.boundary_conds(2) == -1 % if periodic boundary cond
            #    D1x(end, 1) = 1;
            #    D1x(1, end) = 1;

            # D1y = spdiags([ey - 2 * ey ey], [-1 0 1], nPix(2), nPix(2))';
            D1y = sp.coo_matrix.transpose(sp.coo_matrix(
                sp.spdiags(np.concatenate((ey, -2 * ey, ey), axis=0), [-1, 0, 1], stim_dims[2], stim_dims[2])))
            # if stim_params.boundary_conds(3) == -1 % if periodic boundary cond
            #    D1y(end, 1) = 1;
            #    D1y(1, end) = 1;

            Tmat = sp.kron(D1y, sp.kron(Ix, It)) + sp.kron(Iy, sp.kron(D1x, It)) + sp.kron(Iy, sp.kron(Ix, D1t))

        else:
            print('Unsupported reg type.')
            Tmat = None

    Tmat = Tmat.toarray()  # make dense matrix before sending home

    return Tmat


def create_maxpenalty_matrix(input_dims, reg_type):
    """
    Usage: Tmat = create_maxpenalty_matrix(input_dims, reg_type)

    Creates a matrix specifying a an L2-regularization operator of the form
    ||T*k||^2, where T is a matrix and k is a vector of parameters. Currently 
    only supports second derivative/Laplacian operations

    Args:
        input_dims (list of ints): dimensions associated with the target input, 
            in the form [num_lags, num_x_pix, num_y_pix]
        reg_type (str): specify form of the regularization matrix
            'max' | 'max_filt' | 'max_space'

    Returns:
        numpy array: matrix specifying the desired Tikhonov operator

    Notes:
        Adapted from create_Tikhonov_matrix function above.
        
    """

    allowed_reg_types = ['max', 'max_filt', 'max_space', 'center']
    # assert (ischar(reg_type) && ismember(reg_type, allowed_reg_types), 'not an allowed regularization type');

    # first dimension is assumed to represent filters
    num_filt = input_dims[0]

    # additional dimensions are spatial (Nx and Ny)
    num_pix = input_dims[1] * input_dims[2]
    NK = num_filt * num_pix

    rmat = np.zeros([NK, NK], dtype=np.float32)
    if reg_type is 'max':
        # Simply subtract the diagonal from all-ones
        rmat = np.ones([NK, NK], dtype=np.float32) - np.eye(NK, dtype=np.float32)

    elif reg_type is 'max_filt':
        ek = np.ones([num_filt, num_filt], dtype=np.float32) - np.eye(num_filt, dtype=np.float32)
        rmat = np.kron(np.eye(num_pix), ek)

    elif reg_type is 'max_space':
        ex = np.ones([num_pix, num_pix]) - np.eye(num_pix)
        rmat = np.kron(ex, np.eye(num_filt, dtype=np.float32))

    elif reg_type is 'center':
        for i in range(NK):

            pos_x = (i % (input_dims[0] * input_dims[1])) // input_dims[0]
            pos_y = i // (input_dims[0] * input_dims[1])

            center_x = input_dims[1]//2
            center_y = input_dims[2]//2

            alpha = np.square(pos_x - center_x) + np.square(pos_y - center_y)

            rmat[i, i] = 0.01*alpha

    else:
        print('Havent made this type of reg yet. What you are getting wont work.')

    return rmat
