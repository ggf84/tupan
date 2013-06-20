# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import copy
import numpy as np
from .sph import Sphs
from .star import Stars
from .blackhole import Blackholes
from .body import AbstractNbodyMethods
from ..lib.utils.timing import decallmethods, timings


__all__ = ["ParticleSystem"]


def make_common_attrs(cls):
    def make_property(attr, doc):
        @timings
        def fget(self):
            return np.concatenate([getattr(obj, attr)
                                   for obj in self.members.values()])

        @timings
        def fset(self, value):
            try:
                for obj in self.members.values():
                    setattr(obj, attr, value)
            except:
                for obj in self.members.values():
                    setattr(obj, attr, value[:obj.n])
                    value = value[obj.n:]

        return property(fget=fget, fset=fset, fdel=None, doc=doc)
    attrs = ((i[0], cls.__name__+"\'s "+i[2])
             for i in cls.attrs+cls.special_attrs)
    for (attr, doc) in attrs:
        setattr(cls, attr, make_property(attr, doc))
    return cls


@decallmethods(timings)
@make_common_attrs
class ParticleSystem(AbstractNbodyMethods):
    """
    This class holds the particle types in the simulation.
    """
    def __init__(self, nstars=0, nbhs=0, nsphs=0):
        """
        Initializer
        """
        self.members = {cls.__name__.lower(): cls(n)
                        for n, cls in [(nstars, Stars),
                                       (nbhs, Blackholes),
                                       (nsphs, Sphs)] if n}

    #
    # miscellaneous methods
    #
    @property
    def stars(self):
        return self.members["stars"]

    @property
    def blackholes(self):
        return self.members["blackholes"]

    @property
    def sphs(self):
        return self.members["sphs"]

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        fmt = type(self).__name__+"(["
        if self.n:
            for (key, obj) in self.members.items():
                fmt += "\n\t{0},".format('\n\t'.join(str(obj).split('\n')))
            fmt += "\n"
        fmt += "])"
        return fmt

    def __len__(self):
        return sum(len(obj) for obj in self.members.values())

    @property
    def n(self):
        return len(self)

    def append(self, obj):
        if obj.n:
            for (k, v) in obj.members.items():
                if v.n:
                    if not k in self.members:
                        self.members[k] = v.copy()
                    else:
                        self.members[k].append(v)

    def __hash__(self):
        return hash(tuple(self.members.values()))

    def __getitem__(self, slc):
        if isinstance(slc, list):
            slc = np.array(slc)

        if isinstance(slc, np.ndarray):
            if slc.all():
                return self
            subset = type(self)()
            if slc.any():
                for (key, obj) in self.members.items():
                    if obj.n:
                        subset.append(obj[slc[:obj.n]])
                        slc = slc[obj.n:]
            return subset

        if isinstance(slc, int):
            if slc < 0:
                slc = self.n + slc
            if abs(slc) > self.n-1:
                raise IndexError(
                    "index {0} out of bounds 0<=index<{1}".format(slc, self.n))
            subset = type(self)()
            n = 0
            for (key, obj) in self.members.items():
                if obj.n:
                    if n <= slc < n+obj.n:
                        subset.append(obj[slc-n])
                    n += obj.n
            return subset

        if isinstance(slc, slice):
            subset = type(self)()
            start = slc.start
            stop = slc.stop
            if start is None:
                start = 0
            if stop is None:
                stop = self.n
            if start < 0:
                start = self.n + start
            if stop < 0:
                stop = self.n + stop
            for (key, obj) in self.members.items():
                if obj.n:
                    if stop >= 0:
                        if start < obj.n:
                            subset.append(obj[start-obj.n:stop])
                    start -= obj.n
                    stop -= obj.n

            return subset


########## end of file ##########
