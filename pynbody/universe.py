#!/usr/bin/env python
# -*- coding: utf-8 -*-


#"""This module holds base classes for particle types"""

#import numpy as np


#class Universe(dict):
#    """This class holds the particle types in the simulation"""

#    def __new__(cls):
#        """constructor"""
#        return dict.__new__(cls)

#    def __init__(self):
#        """initializer"""
#        dict.__init__(self)

#        # Body
#        self['body'] = np.array([])

#        # BlackHole
#        self['bh'] = np.array([])

##        # Sph
#        self['sph'] = np.array([])

##        # Star
#        self['star'] = np.array([])

#    def set_members(self, member='', data=None):
#        if 'body' in member:
##            tmp = np.ndarray(len(data), dtype=Body)
##            tmp.setfield(data, dtype=Body)
##            self['body']['array'] = np.array(tmp, dtype=Body)
#            self['body'] = data

#        if 'bh' in member:
##            tmp = np.ndarray(len(data), dtype=BlackHole)
##            tmp.setfield(data, dtype=BlackHole)
##            self['bh']['array'] = np.array(tmp, dtype=BlackHole)
#            self['bh'] = data

#        if 'sph' in member:
##            tmp = np.ndarray(len(data), dtype=Sph)
##            tmp.setfield(data, dtype=Sph)
##            self['sph']['array'] = np.array(tmp, dtype=Sph)
#            self['sph'] = data

#        if 'star' in member:
##            tmp = np.ndarray(len(data), dtype=Star)
##            tmp.setfield(data, dtype=Star)
##            self['star']['array'] = np.array(tmp, dtype=Star)
#            self['star'] = data


########## end of file ##########
