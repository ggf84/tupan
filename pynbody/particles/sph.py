#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
from .pbase import Pbase


__all__ = ['Sph']


class Sph(Pbase):
    """
    A base class for Sph.
    """
    dtype = [# common attributes
             ("id", "u8"),
             ("mass", "f8"),
             ("pos", "3f8"),
             ("vel", "3f8"),
             ("acc", "3f8"),
             ("phi", "f8"),
             ("eps2", "f8"),
             ("t_curr", "f8"),
             ("dt_prev", "f8"),
             ("dt_next", "f8"),
             # specific attributes

             # auxiliary attributes

            ]

    def __init__(self, n=0):
        super(Sph, self).__init__(self.dtype, n)


    #
    # specific attributes
    #

    ### ...


    #
    # auxiliary attributes
    #

    ### ...


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
