"""Neural deep network situtation-specific utils by Hadi"""


from __future__ import print_function
from __future__ import division

import numpy as np
#import NDN as NDN
#import NDNutils as NDNutils
from copy import deepcopy



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
