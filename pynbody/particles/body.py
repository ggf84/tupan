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
    #--format--:  (name, type, doc)
    attributes = [# common attributes
                  ('id', 'u8', 'index'),
                  ('mass', 'f8', 'mass'),
                  ('pos', '3f8', 'position'),
                  ('vel', '3f8', 'velocity'),
                  ('acc', '3f8', 'acceleration'),
                  ('phi', 'f8', 'potential'),
                  ('eps2', 'f8', 'softening'),
                  ('t_curr', 'f8', 'current time'),
                  ('dt_prev', 'f8', 'previous time-step'),
                  ('dt_next', 'f8', 'next time-step'),
                  ('nstep', 'u8', 'step number'),
                  # specific attributes
                  ('age', 'f8', 'age'),
                  ('radius', 'f8', 'radius'),
                  ('metallicity', 'f8', 'metallicity'),
                  # auxiliary attributes
                  ('jerk', '3f8', 'a\' = d(a)/dt'),
                 ]

    attrs = ["id", "mass", "pos", "vel", "acc", "phi", "eps2",
             "t_curr", "dt_prev", "dt_next", "nstep", "radius"]

    dtype = [(_[0], _[1]) for _ in attributes]

    zero = np.zeros(0, dtype)


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
