# -*- coding: utf-8 -*-
#

"""
TODO.
"""


import sys
import numpy as np


use_sp = True if "--use_sp" in sys.argv else False
prec = "single" if use_sp else "double"

INT = np.dtype(np.int32)
UINT = np.dtype(np.uint32)
REAL = np.dtype(np.float32) if use_sp else np.dtype(np.float64)

ctypedict = {'int': INT, 'uint': UINT, 'real': REAL}


########## end of file ##########
