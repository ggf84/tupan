# -*- coding: utf-8 -*-
#

"""
TODO.
"""


import sys
import numpy as np


use_sp = True if "--use_sp" in sys.argv else False
prec = "float32" if use_sp else "float64"

INT = np.dtype(np.int32) if use_sp else np.dtype(np.int64)
UINT = np.dtype(np.uint32) if use_sp else np.dtype(np.uint64)
REAL = np.dtype(np.float32) if use_sp else np.dtype(np.float64)

ctypedict = {'int': INT, 'uint': UINT, 'real': REAL}


########## end of file ##########
