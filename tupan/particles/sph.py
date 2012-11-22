#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
from .body import Bodies
from ..lib.utils.timing import decallmethods, timings
from ..lib.utils.dtype import *


__all__ = ["Sphs"]


@decallmethods(timings)
class Sphs(Bodies):
    """

    """
    dtype = Bodies.dtype + [
                            ('rho', REAL),
                           ]


########## end of file ##########
