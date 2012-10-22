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
    special_attrs = [# name, dtype, doc
                     ("rho", "f8", "density"),
                    ]
    special_names = [_[0] for _ in special_attrs]
    special_dtype = [(_[0], _[1]) for _ in special_attrs]
    special_data0 = np.zeros(0, special_dtype) if special_attrs else None

    attrs = Body.attrs + special_attrs
    names = Body.names + special_names
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

from .body import vBody, Bodies, make_properties   # XXX

@decallmethods(timings)
class vSph(vBody):
    """

    """
    def __init__(self):
        super(vSph, self).__init__()
        self.rho = 0.0


@decallmethods(timings)
@make_properties
class Sphs(Bodies):
    """

    """
    basetype = vSph
    dtype = Bodies.dtype + [
                            ('rho', np.float64),
                           ]


########## end of file ##########
