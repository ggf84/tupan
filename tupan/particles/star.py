#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
from .body import Bodies
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Stars"]


@decallmethods(timings)
class Stars(Bodies):
    """

    """
    dtype = Bodies.dtype + [
                            ("sx", np.float64),
                            ("sy", np.float64),
                            ("sz", np.float64),
                            ("radius", np.float64),
                            ("age", np.float64),
                            ("metallicity", np.float64),
                           ]


########## end of file ##########
