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
             ("key", "u8"),
             ("mass", "f8"),
             ("pos", "3f8"),
             ("vel", "3f8"),
             ("acc", "3f8"),
             ("phi", "f8"),
             ("eps2", "f8"),
             ("tcurr", "f8"),
             ("tnext", "f8"),
             # specific attributes

             # auxiliary attributes
             ("tstep", "f8"),
            ]

    def __init__(self, n=0):
        super(Sph, self).__init__(self.dtype, n)


########## end of file ##########
