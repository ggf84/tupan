#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


import sys
import traceback
import numpy as np
from .body import Body
from .blackhole import BlackHole
from .sph import Sph


__all__ = ['Particles']


ALL_PARTICLE_TYPES = ('body', 'blackhole', 'sph')


class Particles(dict):
    """
    This class holds the particle types in the simulation.
    """

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

    def copy(self):
        ret = self.__class__()
        for (key, obj) in self.iteritems():
            if obj:
                ret[key] = obj.copy()
        return ret

    def append(self, data):
        for (key, obj) in data.iteritems():
            if obj:
                if self[key]:
#                    tmp = self[key][:]
                    tmp = self[key].copy()
                    tmp.append(obj)
                    self[key] = tmp
                else:
                    self[key] = obj


    def set_members(self, data):
        """
        Set particle member types.
        """
        try:
            if isinstance(data, Body):
                self['body'] = data
            elif isinstance(data, BlackHole):
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