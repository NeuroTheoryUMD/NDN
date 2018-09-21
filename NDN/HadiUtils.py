"""Neural deep network situtation-specific utils by Hadi"""


from __future__ import print_function
from __future__ import division

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


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
