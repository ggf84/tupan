#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import math
import numpy as np


class Sph(object):
    """A base class for Sph-type particles"""

    def __init__(self):
        self.array = np.array([],
                              dtype=[('index', 'u8'), ('nstep', 'u8'),
                                     ('tstep', 'f8'), ('time', 'f8'),
                                     ('mass', 'f8'), ('eps2', 'f8'), ('pot', 'f8'),
                                     ('rho', 'f8'), ('press', 'f8'), ('temp', 'f8'),
                                     ('pos', 'f8', (3,)), ('vel', 'f8', (3,))])

        # total mass
        self.total_mass = 0.0

    def __repr__(self):
        return '{array}'.format(array=self.array)

    def __len__(self):
        return len(self.array)

    def get_data(self):
        return self.array

    def fromlist(self, data):
        self.array = np.asarray(data, dtype=self.array.dtype)
        self.total_mass = np.sum(self.array['mass'])




########## end of file ##########
