#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
from .body import Bodies
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Sphs"]


@decallmethods(timings)
class Sphs(Bodies):
    """

    """
    dtype = Bodies.dtype + [
                            ('rho', np.float64),
                           ]


########## end of file ##########
