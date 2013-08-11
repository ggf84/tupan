# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
from .body import Bodies
from ..lib.utils.timing import decallmethods, timings
from ..lib.utils import ctype


__all__ = ["Blackholes"]


@decallmethods(timings)
class Blackholes(Bodies):
    """

    """
    attrs = Bodies.attrs + [
        ("spinx", ctype.REAL, "x-spin"),
        ("spiny", ctype.REAL, "y-spin"),
        ("spinz", ctype.REAL, "z-spin"),
        ("radius", ctype.REAL, "radius"),
    ]
    dtype = [(_[0], _[1]) for _ in attrs]


########## end of file ##########
