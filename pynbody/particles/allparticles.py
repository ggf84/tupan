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
        @timings
        def fget(self):
            return np.concatenate([getattr(obj, attr) for obj in self.values()])

        @timings
        def fset(self, value):
            try:
                for obj in self.values():
                    setattr(obj, attr, value)
            except:
                for obj in self.values():
                    setattr(obj, attr, value[:obj.n])
                    value = value[obj.n:]

        return property(fget=fget, fset=fset, fdel=None, doc=doc)
    attrs = ((i[0], cls.__name__+"\'s "+i[2]) for i in cls.attrs+cls.special_attrs)
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

#        self.kind["star"] = Star(nstar)
#        self.n += nstar
#
#        self.kind["blackhole"] = BlackHole(nbh)
#        self.n += nbh
#
#        self.kind["sph"] = Sph(nsph)
#        self.n += nsph


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
        return hash(tuple(self.values()))


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
                    if not k in self.kind:
                        self.kind[k] = v
                    else:
                        self.kind[k].append(v)
            self.n = len(self)



###############################################################################

from .sph import Sphs
from .star import Stars
from .blackhole import Blackholes
from .body import Bodies

#Particles = Bodies

@decallmethods(timings)
class Particles_(object):    # XXX: rename -> System
    """
    This class holds the particle types in the simulation.
    """
#    def __init__(self, nstar=0, nbh=0, nsph=0, items=None):
#        """
#        Initializer
#        """
#        items = {cls.__name__.lower(): cls(n) for n, cls in [(nstar, Stars),
#                                                             (nbh, Blackholes),
#                                                             (nsph, Sphs)]}
#        self.__dict__.update(items)

    def __repr__(self):
        return str(self.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()
    members = values

    def items(self):
        return self.__dict__.items()

    def __len__(self):
        return sum([obj.n for obj in self.__dict__.values()])
    n = property(__len__)

    @property
    def total_mass(self):
        """
        Total mass.
        """
        return sum([obj.total_mass for obj in self.values() if obj.n])


    @property
    def rcom(self):
        """
        Position of the center-of-mass.
        """
        mtot = self.total_mass
        rcom = np.array([0.0, 0.0, 0.0])
        for obj in self.values():
            if obj.n:
                rcom += obj.rcom * obj.total_mass
        return rcom / mtot


    @property
    def vcom(self):
        """
        Velocity of the center-of-mass.
        """
        mtot = self.total_mass
        vcom = np.array([0.0, 0.0, 0.0])
        for obj in self.values():
            if obj.n:
                vcom += obj.vcom * obj.total_mass
        return vcom / mtot


    @property
    def linear_momentum(self):
        """
        Total linear momentum.
        """
        linear_momentum = np.array([0.0, 0.0, 0.0])
        for obj in self.values():
            if obj.n:
                linear_momentum += obj.linear_momentum
        return linear_momentum


    @property
    def angular_momentum(self):
        """
        Total angular momentum.
        """
        angular_momentum = np.array([0.0, 0.0, 0.0])
        for obj in self.values():
            if obj.n:
                angular_momentum += obj.angular_momentum
        return angular_momentum


    @property
    def kinetic_energy(self):
        """
        Total kinetic energy.
        """
        return sum([obj.kinetic_energy for obj in self.values() if obj.n])


    @property
    def potential_energy(self):
        """
        Total potential energy.
        """
        return sum([obj.potential_energy for obj in self.values() if obj.n])


    @property
    def virial_energy(self):
        """
        Total virial energy.
        """
        return sum([obj.virial_energy for obj in self.values() if obj.n])


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
                    self.__dict__.setdefault(k, type(v)()).append(v)


    def as_body(self):
        b = Bodies()
        for obj in self.values():
            if obj.n:
                b.append(obj.astype(Bodies))
        return b

    def update_tstep(self, objs, eta):
        """
        Update the individual time-steps due to other particles.
        """
        bodies = objs.as_body()
        [obj.update_tstep(bodies, eta) for obj in self.values() if obj.n]


########## end of file ##########
