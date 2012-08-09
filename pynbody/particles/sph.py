#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
from .pbase import Pbase, make_attrs
from ..lib.utils.timing import decallmethods, timings


__all__ = ['Sph']


@decallmethods(timings)
@make_attrs
class Sph(Pbase):
    """
    A base class for Sph.
    """
    specific_attrs = [# name, dtype, doc

                     ]

    specific_names = [_[0] for _ in specific_attrs]

    attrs = Pbase.common_attrs + specific_attrs
    names = Pbase.common_names + specific_names

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
