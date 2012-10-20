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
    special_attrs = [# name, dtype, doc
                     ("spin", "3f8", "spin"),
                     ("radius", "f8", "radius"),
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

class vBlackhole(vBody):
    """

    """
    def __init__(self, _id):
        super(vBlackhole, self).__init__(_id)
        self.rad = 0.0
        self.pnaccx = 0.0
        self.pnaccy = 0.0
        self.pnaccz = 0.0


@make_properties
class vBlackholes(vBodies):
    """

    """
    dtype = vBodies.dtype + [('rad', np.float64),
                             ('pnaccx', np.float64),
                             ('pnaccy', np.float64),
                             ('pnaccx', np.float64), ]
    def __init__(self, n=0, objs=None):
        if n: self.objs = np.vectorize(vBlackhole)(xrange(n))
        elif objs is not None: self.objs = objs
        else: self.objs = np.zeros(n, object)


########## end of file ##########
