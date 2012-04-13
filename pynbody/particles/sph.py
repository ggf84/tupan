#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
from collections import namedtuple
#from collections import (namedtuple, OrderedDict)
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









###############################################################################
### XXX: old Sph



dtype = {"names":   ["index", "mass", "eps2", "phi", "rho", "press", "temp", "pos", "vel", "acc"],
         "formats": ["u8",    "f8",   "f8",   "f8",  "f8",  "f8",    "f8",   "3f8", "3f8", "3f8"]}

#fields = OrderedDict([('index', 'u8'), ('mass', 'f8'), ('eps2', 'f8'),
#                      ('phi', 'f8'), ('rho', 'f8'), ('press', 'f8'),
#                      ('temp', 'f8'), ('pos', '3f8'),
#                      ('vel', '3f8'), ('acc', '3f8')])
##dtype = fields.items()
#dtype = {'names': fields.keys(), 'formats': fields.values()}


Energies = namedtuple("Energies", ["kin", "pot", "tot", "vir"])


class oldSph(Pbase):
    """
    A base class for Sph-type particles
    """
    def __init__(self, numobjs=0):
        Pbase.__init__(self, numobjs, dtype)



########## end of file ##########
