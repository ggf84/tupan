# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import numpy as np
from ...config import options


class Ctype(object):
    int_t = (np.int32
             if options.fpwidth == 'fp32'
             else (np.int64
                   if options.fpwidth == 'fp64'
                   else np.int64))

    uint_t = (np.uint32
              if options.fpwidth == 'fp32'
              else (np.uint64
                    if options.fpwidth == 'fp64'
                    else np.uint64))

    real_t = (np.float32
              if options.fpwidth == 'fp32'
              else (np.float64
                    if options.fpwidth == 'fp64'
                    else np.float64))

    c_int = ('int'
             if options.fpwidth == 'fp32'
             else ('long'
                   if options.fpwidth == 'fp64'
                   else 'long'))

    c_uint = ('unsigned int'
              if options.fpwidth == 'fp32'
              else ('unsigned long'
                    if options.fpwidth == 'fp64'
                    else 'unsigned long'))

    c_real = ('float'
              if options.fpwidth == 'fp32'
              else ('double'
                    if options.fpwidth == 'fp64'
                    else 'double'))


# -- End of File --
