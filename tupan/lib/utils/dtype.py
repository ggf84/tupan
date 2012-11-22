#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

import sys
import numpy as np


__all__ = ["UINT", "REAL"]


UINT = np.uint32
REAL = np.float32 if "--use_sp" in sys.argv else np.float64


########## end of file ##########
