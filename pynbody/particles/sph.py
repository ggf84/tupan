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

from .body import vBody, vBodies, make_properties   # XXX

class vSph(vBody):
    """

    """
    def __init__(self, _id):
        super(vSph, self).__init__(_id)
        self.rho = 0.0


@make_properties
class vSphs(vBodies):
    """

    """
    dtype = vBodies.dtype + [('rho', np.float64), ]
    def __init__(self, n=0, objs=None):
        if n: self.objs = np.vectorize(vSph)(xrange(n))
        elif objs is not None: self.objs = objs
        else: self.objs = np.zeros(n, object)


########## end of file ##########
