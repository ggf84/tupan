#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
from .body import Bodies
from ..lib.utils.timing import decallmethods, timings
from ..lib.utils.dtype import *


__all__ = ["Blackholes"]


@decallmethods(timings)
class Blackholes(Bodies):
    """

    """
    dtype = Bodies.dtype + [
                            ("sx", REAL),
                            ("sy", REAL),
                            ("sz", REAL),
                            ("radius", REAL),
                           ]


########## end of file ##########
