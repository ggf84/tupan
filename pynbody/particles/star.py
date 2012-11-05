#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
from .body import Body, make_attrs
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Star"]


@decallmethods(timings)
@make_attrs
class Star(Body):
    """
    A base class for Stars.
    """
    special_attrs = [# name, dtype, doc
                     ("sx", "f8", "x-spin"),
                     ("sy", "f8", "y-spin"),
                     ("sz", "f8", "z-spin"),
                     ("radius", "f8", "radius"),
                     ("age", "f8", "age"),
                     ("metallicity", "f8", "metallicity"),
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
class vStar(vBody):
    """

    """
    def __init__(self):
        super(vStar, self).__init__()
        self.sx = 0.0
        self.sy = 0.0
        self.sz = 0.0
        self.radius = 0.0
        self.age = 0.0
        self.metallicity = 0.0



@decallmethods(timings)
#@make_properties
class Stars(Bodies):
    """

    """
#    basetype = vStar
#    dtype = Bodies.dtype + [
#                            ("sx", np.float64),
#                            ("sy", np.float64),
#                            ("sz", np.float64),
#                            ("radius", np.float64),
#                            ("age", np.float64),
#                            ("metallicity", np.float64),
#                           ]


########## end of file ##########
