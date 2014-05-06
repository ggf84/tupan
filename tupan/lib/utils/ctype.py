# -*- coding: utf-8 -*-
#

"""
TODO.
"""


import sys
import numpy as np


class Ctype(object):
    fpwidth = 'fp32' if "--use_sp" in sys.argv else 'fp64'

    int = (np.dtype(np.int32)
           if fpwidth == 'fp32'
           else (np.dtype(np.int64)
                 if fpwidth == 'fp64'
                 else np.dtype(np.int64)
                 )
           )

    uint = (np.dtype(np.uint32)
            if fpwidth == 'fp32'
            else (np.dtype(np.uint64)
                  if fpwidth == 'fp64'
                  else np.dtype(np.uint64)
                  )
            )

    real = (np.dtype(np.float32)
            if fpwidth == 'fp32'
            else (np.dtype(np.float64)
                  if fpwidth == 'fp64'
                  else np.dtype(np.float64)
                  )
            )


# -- End of File --
