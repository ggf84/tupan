# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from .body import Bodies
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Blackholes"]


@decallmethods(timings)
class Blackholes(Bodies):
    """

    """
    attrs = Bodies.attrs + [
        ("spinx", "real", "x-spin"),
        ("spiny", "real", "y-spin"),
        ("spinz", "real", "z-spin"),
        ("radius", "real", "radius"),
    ]


########## end of file ##########
