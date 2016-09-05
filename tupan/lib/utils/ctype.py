# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import numpy as np
from ...config import cli


class Ctype(object):
    int_t = (np.int32
             if cli.fpwidth == 'fp32'
             else (np.int64
                   if cli.fpwidth == 'fp64'
                   else np.int64))

    uint_t = (np.uint32
              if cli.fpwidth == 'fp32'
              else (np.uint64
                    if cli.fpwidth == 'fp64'
                    else np.uint64))

    real_t = (np.float32
              if cli.fpwidth == 'fp32'
              else (np.float64
                    if cli.fpwidth == 'fp64'
                    else np.float64))

    c_int = ('int'
             if cli.fpwidth == 'fp32'
             else ('long'
                   if cli.fpwidth == 'fp64'
                   else 'long'))

    c_uint = ('unsigned int'
              if cli.fpwidth == 'fp32'
              else ('unsigned long'
                    if cli.fpwidth == 'fp64'
                    else 'unsigned long'))

    c_real = ('float'
              if cli.fpwidth == 'fp32'
              else ('double'
                    if cli.fpwidth == 'fp64'
                    else 'double'))


# -- End of File --
