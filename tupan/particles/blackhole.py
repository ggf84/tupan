# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
from .body import Bodies
from ..lib.utils.timing import decallmethods, timings
from ..lib.utils.dtype import *


__all__ = ["Blackholes"]


@decallmethods(timings)
class Blackholes(Bodies):
    """

    """
    attrs = Bodies.attrs + [
        ("sx", REAL, "x-spin"),
        ("sy", REAL, "y-spin"),
        ("sz", REAL, "z-spin"),
        ("radius", REAL, "radius"),
    ]
    dtype = [(_[0], _[1]) for _ in attrs]


########## end of file ##########
