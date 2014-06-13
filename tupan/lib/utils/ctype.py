# -*- coding: utf-8 -*-
#

"""
TODO.
"""


import numpy as np
from tupan.config import options


class Ctype(object):
    int = (np.dtype(np.int32)
           if options.fpwidth == 'fp32'
           else (np.dtype(np.int64)
                 if options.fpwidth == 'fp64'
                 else np.dtype(np.int64)))

    uint = (np.dtype(np.uint32)
            if options.fpwidth == 'fp32'
            else (np.dtype(np.uint64)
                  if options.fpwidth == 'fp64'
                  else np.dtype(np.uint64)))

    real = (np.dtype(np.float32)
            if options.fpwidth == 'fp32'
            else (np.dtype(np.float64)
                  if options.fpwidth == 'fp64'
                  else np.dtype(np.float64)))


# -- End of File --
