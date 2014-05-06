# -*- coding: utf-8 -*-
#

"""
TODO.
"""


import numpy as np
from .body import Bodies, typed_property
from ..lib.utils.timing import timings, bind_all
from ..lib.utils.ctype import Ctype


__all__ = ["Sphs"]


@bind_all(timings)
class Sphs(Bodies):
    """

    """
    density = typed_property('density',
                             lambda self: np.zeros(self.n, dtype=Ctype.real))

    attrs = Bodies.attrs + [
        ('density', "real", "density"),
    ]


# -- End of File --
