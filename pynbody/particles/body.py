#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
from .pbase import Pbase


__all__ = ["Body"]


class Body(Pbase):
    """
    A base class for Stars.
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
             ("age", "f8"),
             ("radius", "f8"),
             ("metallicity", "f8"),
             # auxiliary attributes
             ("tstep", "f8"),
            ]

    def __init__(self, n=0):
        super(Body, self).__init__(self.dtype, n)


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

    ### tstep

    @property
    def tstep(self):
        return self.data['tstep']

    @tstep.setter
    def tstep(self, values):
        self.data['tstep'] = values

    @tstep.deleter
    def tstep(self):
        raise NotImplementedError()


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
