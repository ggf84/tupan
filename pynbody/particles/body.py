#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
from .pbase import Pbase, make_attrs
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Body"]


@decallmethods(timings)
@make_attrs
class Body(Pbase):
    """
    A base class for Stars.
    """
    special_attrs = [# name, dtype, doc
                     ("age", "f8", "age"),
                     ("radius", "f8", "radius"),
                     ("metallicity", "f8", "metallicity"),
                    ]
    special_names = [_[0] for _ in special_attrs]
    special_dtype = [(_[0], _[1]) for _ in special_attrs]
    special_data0 = np.zeros(0, special_dtype) if special_attrs else None

    attrs = Pbase.common_attrs + special_attrs
    names = Pbase.common_names + special_names
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
