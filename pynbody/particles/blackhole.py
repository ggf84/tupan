#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
from .body import Body, make_attrs
from ..lib import gravity
from ..lib.utils.memoize import cache, cache_arg
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


########## end of file ##########
