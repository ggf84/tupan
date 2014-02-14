# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from .body import Bodies
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Stars"]


@decallmethods(timings)
class Stars(Bodies):
    """

    """
    attrs = Bodies.attrs + [
        ("spinx", "real", "x-spin"),
        ("spiny", "real", "y-spin"),
        ("spinz", "real", "z-spin"),
        ("radius", "real", "radius"),
        ("age", "real", "age"),
        ("metallicity", "real", "metallicity"),
    ]


########## end of file ##########
