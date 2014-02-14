# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from .body import Bodies
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Sphs"]


@decallmethods(timings)
class Sphs(Bodies):
    """

    """
    attrs = Bodies.attrs + [
        ('density', "real", "density"),
    ]


########## end of file ##########
