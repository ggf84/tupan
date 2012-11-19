#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
from .body import Body, make_attrs
from ..lib.utils.timing import decallmethods, timings


__all__ = ["BlackHole"]


@decallmethods(timings)
@make_attrs
class BlackHole(Body):
    """
    A base class for BlackHoles.
    """
    attrs = Body.attrs + [# name, dtype, doc
                          ("sx", "f8", "x-spin"),
                          ("sy", "f8", "y-spin"),
                          ("sz", "f8", "z-spin"),
                          ("radius", "f8", "radius"),
                         ]
    names = Body.names + [_[0] for _ in attrs]
    dtype = [(_[0], _[1]) for _ in attrs]
    data0 = np.zeros(0, dtype)


    #
    # specific methods
    #

    ### ...


    #
    # auxiliary methods
    #

    ### ...


    #
    # overridden methods
    #

    ### ...




###############################################################################

from .body import Bodies

@decallmethods(timings)
class Blackholes(Bodies):
    """

    """
    dtype = Bodies.dtype + [
                            ("sx", np.float64),
                            ("sy", np.float64),
                            ("sz", np.float64),
                            ("radius", np.float64),
                           ]


########## end of file ##########
