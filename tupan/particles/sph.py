# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
from .body import Bodies
from ..lib.utils.timing import decallmethods, timings
from ..lib.utils import ctype


__all__ = ["Sphs"]


@decallmethods(timings)
class Sphs(Bodies):
    """

    """
    attrs = Bodies.attrs + [
        ('rho', ctype.REAL, "density"),
    ]
    dtype = [(_[0], _[1]) for _ in attrs]


########## end of file ##########
