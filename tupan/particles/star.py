#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
from .body import Bodies
from ..lib.utils.timing import decallmethods, timings
from ..lib.utils.dtype import *


__all__ = ["Stars"]


@decallmethods(timings)
class Stars(Bodies):
    """

    """
    attrs = Bodies.attrs + [
                            ("sx", REAL, "x-spin"),
                            ("sy", REAL, "y-spin"),
                            ("sz", REAL, "z-spin"),
                            ("radius", REAL, "radius"),
                            ("age", REAL, "age"),
                            ("metallicity", REAL, "metallicity"),
                           ]
    dtype = [(_[0], _[1]) for _ in attrs]


########## end of file ##########
