#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
from .pbase import Pbase
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Body"]


@decallmethods(timings)
class Body(Pbase):
    """
    A base class for Stars.
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
             ("age", "f8"),
             ("radius", "f8"),
             ("metallicity", "f8"),
             # auxiliary attributes

            ]

    zero = np.zeros(0, dtype)

    def __init__(self, n=0):
        super(Body, self).__init__(n)


    #
    # specific attributes
    #

    ### age

    @property
    def age(self):
        return self.data['age']

    @age.setter
    def age(self, values):
        self.data['age'] = values

    @age.deleter
    def age(self):
        raise NotImplementedError()


    ### radius

    @property
    def radius(self):
        return self.data['radius']

    @radius.setter
    def radius(self, values):
        self.data['radius'] = values

    @radius.deleter
    def radius(self):
        raise NotImplementedError()


    ### metallicity

    @property
    def metallicity(self):
        return self.data['metallicity']

    @metallicity.setter
    def metallicity(self, values):
        self.data['metallicity'] = values

    @metallicity.deleter
    def metallicity(self):
        raise NotImplementedError()


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
