#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import print_function
import sys
import copy
import numpy as np
from .sph import Sph
from .star import Star
from .blackhole import BlackHole
from .body import AbstractNbodyMethods
from ..lib import gravity
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Particles"]


ALL_PARTICLE_TYPES = ["sph", "star", "blackhole"]


def make_common_attrs(cls):
    def make_property(attr, doc):
        def fget(self):
            seq = [obj.data[attr] for obj in self.objs() if obj.n]
            if len(seq) == 1:
                return seq[0]
            if len(seq) > 1:
                return np.concatenate(seq)
            return np.concatenate([obj.data[attr] for obj in self.objs()])

#            seq = [v for obj in self.objs() for v in obj.data[attr]]
#            return np.array(seq)


        def fset(self, value):
            for obj in self.objs():
                if obj.n:
                    try:
                        obj.data[attr] = value[:obj.n]
                        value = value[obj.n:]
                    except:
                        obj.data[attr] = value

#            for obj in self.objs():
#                if obj.n:
#                    try:
#                        items = value[:obj.n]
#                        value = value[obj.n:]
#                    except:
#                        items = [value]*obj.n
#                    for i, v in enumerate(items):
#                        obj.data[attr][i] = v


        def fdel(self):
            raise NotImplementedError()
        return property(fget, fset, fdel, doc)
    attrs = ((i[0], cls.__name__+"\'s "+i[2]) for i in cls.attrs)
    for (attr, doc) in attrs:
        setattr(cls, attr, make_property(attr, doc))
    return cls



@decallmethods(timings)
@make_common_attrs
class Particles(AbstractNbodyMethods):
    """
    This class holds the particle types in the simulation.
    """
    def __init__(self, nstar=0, nbh=0, nsph=0):
        """
        Initializer
        """
        self.kind = {}
        self.n = 0

        self.kind["star"] = Star(nstar)
        self.n += nstar

        self.kind["blackhole"] = BlackHole(nbh)
        self.n += nbh

        self.kind["sph"] = Sph(nsph)
        self.n += nsph


    @property
    def star(self):
        return self.kind["star"]

    @property
    def blackhole(self):
        return self.kind["blackhole"]

    @property
    def sph(self):
        return self.kind["sph"]


    #
    # miscellaneous methods
    #

    def __str__(self):
        fmt = type(self).__name__+"([\n"
        for (key, obj) in self.items():
            fmt += "{0},\n".format(obj)
        fmt += "])"
        return fmt


    def __repr__(self):
        return str(self.kind)


    def __hash__(self):
        return hash(tuple(self.objs()))


    def __getitem__(self, slc):
        if isinstance(slc, int):
            if slc < 0: slc = self.n + slc
            if abs(slc) > self.n-1:
                raise IndexError("index {0} out of bounds 0<=index<{1}".format(slc, self.n))
            subset = type(self)()
            n = 0
            for (key, obj) in self.items():
                if obj.n:
                    if n <= slc < n+obj.n:
                        subset.append(obj[slc-n])
                    n += obj.n
            return subset

        if isinstance(slc, slice):
            subset = type(self)()
            start = slc.start
            stop = slc.stop
            if start is None: start = 0
            if stop is None: stop = self.n
            if start < 0: start = self.n + start
            if stop < 0: stop = self.n + stop
            for (key, obj) in self.items():
                if obj.n:
                    if stop >= 0:
                        if start < obj.n:
                            subset.append(obj[start-obj.n:stop])
                    start -= obj.n
                    stop -= obj.n

            return subset

        if isinstance(slc, list):
            slc = np.array(slc)

        if isinstance(slc, np.ndarray):
            if slc.all():
                return self
            subset = type(self)()
            if slc.any():
                for (key, obj) in self.items():
                    if obj.n:
                        subset.append(obj[slc[:obj.n]])
                        slc = slc[obj.n:]
            return subset


    def append(self, obj):
        if obj.n:
            for (k, v) in obj.items():
                if v.n:
                    self.kind[k].append(v)
            self.n = len(self)


########## end of file ##########
