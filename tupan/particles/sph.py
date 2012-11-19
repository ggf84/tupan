#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
from .body import Body, make_attrs
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Sph"]


@decallmethods(timings)
@make_attrs
class Sph(Body):
    """
    A base class for Sph.
    """
    attrs = Body.attrs + [# name, dtype, doc
                          ("rho", "f8", "density"),
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
class Sphs(Bodies):
    """

    """
    dtype = Bodies.dtype + [
                            ('rho', np.float64),
                           ]


########## end of file ##########
