# -*- coding: utf-8 -*-
#

"""
TODO.
"""


import sys
import numpy as np


use_sp = True if "--use_sp" in sys.argv else False
prec = "single" if use_sp else "double"

INT = np.int32
UINT = np.uint32
REAL = np.float32 if use_sp else np.float64


########## end of file ##########
