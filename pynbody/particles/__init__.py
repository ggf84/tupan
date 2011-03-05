#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Particles

This package implements base classes for particle types in the simulation.
"""

import sys
import traceback
import numpy as np
from .body import Bodies
from .blackhole import BlackHoles
from .sph import Sph


ALL_PARTICLE_TYPES = ('body', 'blackhole', 'sph')


class Particles(dict):
    """
    This class holds the particle types in the simulation.
    """

    def __new__(cls):
        """
        Constructor
        """
        return dict.__new__(cls)

    def __init__(self):
        """
        Initializer
        """
        dict.__init__(self)

        self['body'] = None
        self['blackhole'] = None
        self['sph'] = None


    def any(self):
        has_obj = False
        for obj in self.itervalues():
            if obj:
                has_obj = True
        return has_obj


    def append(self, data):
        for (key, obj) in data.iteritems():
            if obj:
                if self[key]:
                    self[key] = self[key].concatenate(obj)
                else:
                    self[key] = obj


    def set_members(self, data):
        """
        Set particle member types.
        """
        try:
            if isinstance(data, Bodies):
                self['body'] = data
            elif isinstance(data, BlackHoles):
                self['blackhole'] = data
            elif isinstance(data, Sph):
                self['sph'] = data
            else:
                raise TypeError('Unsupported particle type.')
        except TypeError as msg:
            traceback.print_list(
                    traceback.extract_stack(
                            sys.exc_info()[2].tb_frame.f_back, None), None)
            print('TypeError: {0}'.format(msg))
            sys.exit(-1)


########## end of file ##########
