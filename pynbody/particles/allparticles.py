#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import print_function
import sys
import copy
import numpy as np
from .sph import Sph
from .body import Body
from .blackhole import BlackHole
from .pbase import AbstractNbodyMethods
from ..lib import gravity
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Particles"]


ALL_PARTICLE_TYPES = ["sph", "body", "blackhole"]


def make_common_attrs(cls):
    def make_property(attr, doc):
        def fget(self):
            seq = [obj.data[attr] for obj in self.objs() if obj.n]
            if len(seq) == 1:
                return seq[0]
            if len(seq) > 1:
                return np.concatenate(seq)
            return np.concatenate([obj.data[attr] for obj in self.objs()])
        def fset(self, value):
            for obj in self.objs():
                if obj.n:
                    try:
                        obj.data[attr] = value[:obj.n]
                        value = value[obj.n:]
                    except:
                        obj.data[attr] = value
        def fdel(self):
            raise NotImplementedError()
        return property(fget, fset, fdel, doc)
    attrs = ((i[0], cls.__name__+"\'s "+i[2]) for i in cls.common_attrs)
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

        self.kind["body"] = Body(nstar)
        self.n += nstar

        self.kind["blackhole"] = BlackHole(nbh)
        self.n += nbh

        self.kind["sph"] = Sph(nsph)
        self.n += nsph


    @property
    def body(self):
        return self.kind["body"]

    @property
    def blackhole(self):
        return self.kind["blackhole"]

    @property
    def sph(self):
        return self.kind["sph"]

    def keys(self):
        return self.kind.keys()

    def objs(self):
        return self.kind.values()

    def items(self):
        return self.kind.items()


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
        return str(dict(self.items()))


    def __hash__(self):
        i = None
        for obj in self.objs():
            if i is None:
                i = hash(obj)
            else:
                i ^= hash(obj)
        return i


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
            raise NotImplementedError()

        if isinstance(slc, np.ndarray):
            if slc.all():
                return self
            if slc.any():
                subset = type(self)()
                for (key, obj) in self.items():
                    if obj.n:
                        subset.append(obj[slc[:obj.n]])
                        slc = slc[obj.n:]
                return subset
            return type(self)()



    def append(self, objs):
        if isinstance(objs, Particles):
            if objs.n:
                for (key, obj) in objs.items():
                    if obj.n:
                        self.kind[key].append(obj)
                self.n = len(self)
        elif isinstance(objs, (Body, BlackHole, Sph)):
            if objs.n:
                key = type(objs).__name__.lower()
                self.kind[key].append(objs)
                self.n = len(self)
        else:
            raise TypeError("'{}' can not append obj of type: '{}'".format(type(self).__name__, type(objs)))



    #
    # uncommon methods
    #

    ### total mass and center-of-mass

    def get_total_rcom_pn_shift(self):
        """

        """
        mtot = self.total_mass
        rcom_shift = 0.0
        for obj in self.objs():
            if obj.n:
                if hasattr(obj, "get_rcom_pn_shift"):
                    rcom_shift += obj.get_rcom_pn_shift()
        return (rcom_shift / mtot)

    def get_total_vcom_pn_shift(self):
        """

        """
        mtot = self.total_mass
        vcom_shift = 0.0
        for obj in self.objs():
            if obj.n:
                if hasattr(obj, "get_vcom_pn_shift"):
                    vcom_shift += obj.get_vcom_pn_shift()
        return (vcom_shift / mtot)


    ### linear momentum

    def get_total_lmom_pn_shift(self):
        """

        """
        lmom_shift = 0.0
        for obj in self.objs():
            if obj.n:
                if hasattr(obj, "get_lmom_pn_shift"):
                    lmom_shift += obj.get_lmom_pn_shift()
        return lmom_shift


    ### angular momentum

    def get_total_amom_pn_shift(self):
        """

        """
        amom_shift = 0.0
        for obj in self.objs():
            if obj.n:
                if hasattr(obj, "get_amom_pn_shift"):
                    amom_shift += obj.get_amom_pn_shift()
        return amom_shift


    ### kinetic energy

    def get_total_ke_pn_shift(self):
        """

        """
        ke_shift = 0.0
        for obj in self.objs():
            if obj.n:
                if hasattr(obj, "get_ke_pn_shift"):
                    ke_shift += obj.get_ke_pn_shift()
        return ke_shift


    ### gravity

    def update_pnacc(self, objs, pn_order, clight):
        """
        Update the individual post-newtonian gravitational acceleration due to other particles.
        """
        ni = self.blackhole.n
        nj = objs.blackhole.n
        if ni and nj:
            self.blackhole.update_pnacc(objs.blackhole, pn_order, clight)


########## end of file ##########
