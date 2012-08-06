#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
from .pbase import Pbase
from ..lib.utils.timing import decallmethods, timings


__all__ = ['Sph']


@decallmethods(timings)
class Sph(Pbase):
    """
    A base class for Sph.
    """
    specific_attributes = [# name, dtype, doc

                          ]

    specific_names = [_[0] for _ in specific_attributes]

    attributes = Pbase.common_attributes + specific_attributes
    names = Pbase.common_names + specific_names

    dtype = [(_[0], _[1]) for _ in attributes]
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
