# -*- coding: utf-8 -*-
#

"""
TODO.
"""


import numpy as np
from .body import Bodies, typed_property
from ..lib.utils.timing import timings, bind_all
from ..lib.utils.ctype import Ctype


__all__ = ["Stars"]


@bind_all(timings)
class Stars(Bodies):
    """

    """
    spinx = typed_property('spinx',
                           lambda self: np.zeros(self.n, dtype=Ctype.real))
    spiny = typed_property('spiny',
                           lambda self: np.zeros(self.n, dtype=Ctype.real))
    spinz = typed_property('spinz',
                           lambda self: np.zeros(self.n, dtype=Ctype.real))
    radius = typed_property('radius',
                            lambda self: np.zeros(self.n, dtype=Ctype.real))
    age = typed_property('age',
                         lambda self: np.zeros(self.n, dtype=Ctype.real))
    metallicity = typed_property('metallicity',
                                 lambda self: np.zeros(self.n,
                                                       dtype=Ctype.real))

    attrs = Bodies.attrs + [
        ("spinx", "real", "x-spin"),
        ("spiny", "real", "y-spin"),
        ("spinz", "real", "z-spin"),
        ("radius", "real", "radius"),
        ("age", "real", "age"),
        ("metallicity", "real", "metallicity"),
    ]


# -- End of File --
