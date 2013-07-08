# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import copy
import numpy as np
from collections import defaultdict
from .body import Bodies
from .sph import Sphs
from .star import Stars
from .blackhole import Blackholes
from .body import AbstractNbodyMethods
from ..lib.utils.timing import decallmethods, timings


__all__ = ["ParticleSystem"]


@decallmethods(timings)
class ParticleSystem(AbstractNbodyMethods):
    """
    This class holds the particle types in the simulation.
    """
    def __init__(self, nbodies=0, nstars=0, nbhs=0, nsphs=0, members=None):
        """
        Initializer
        """
        if members is None:
            members = {cls.__name__.lower(): cls(n)
                       for (n, cls) in [(nbodies, Bodies),
                                        (nstars, Stars),
                                        (nbhs, Blackholes),
                                        (nsphs, Sphs)] if n}
        self.members = members
        self.n = len(self)
        if self.n:
            self._rebind_attrs()

    def _rebind_attrs(self):
        attrs = defaultdict(list)
        for (key, obj) in self.members.items():
            for attr in obj.__dict__:
                attrs[attr].append(getattr(obj, attr))

        for (attr, seq) in attrs.items():
            setattr(self, attr, np.concatenate(seq))

        ns = 0
        nf = 0
        for (key, obj) in self.members.items():
            setattr(self, key, obj)
            nf += obj.n
            for attr in obj.__dict__:
                setattr(obj, attr, getattr(self, attr)[ns:nf])
            ns += obj.n

    #
    # miscellaneous methods
    #

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

    def copy(self):
        return copy.deepcopy(self)

    def append(self, obj):
        if obj.n:
            try:
                for (k, v) in obj.members.items():
                    try:
                        self.members[k].append(v)
                    except:
                        self.members[k] = v.copy()
            except:
                k, v = type(obj).__name__.lower(), obj
                try:
                    self.members[k].append(v)
                except:
                    self.members[k] = v.copy()
            self.n = len(self)
            self._rebind_attrs()

    def __getitem__(self, slc):
        if isinstance(slc, (list, np.ndarray)):
            if all(slc):
                return self
            if any(slc):
                ns = 0
                nf = 0
                members = {}
                for (key, obj) in self.members.items():
                    nf += obj.n
                    members[key] = obj[slc[ns:nf]]
                    ns += obj.n
                return type(self)(members=members)
            return type(self)()

        if isinstance(slc, int):
            if abs(slc) > self.n-1:
                raise IndexError(
                    "index {0} out of bounds 0<=index<{1}".format(slc, self.n))
            if slc < 0:
                slc = self.n + slc
            n = 0
            members = {}
            for (key, obj) in self.members.items():
                i = slc - n
                if 0 <= i < obj.n:
                    members[key] = obj[i]
                n += obj.n
            return type(self)(members=members)

        if isinstance(slc, slice):
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
            members = {}
            for (key, obj) in self.members.items():
                if obj.n:
                    if stop >= 0:
                        if start < obj.n:
                            members[key] = obj[start-obj.n:stop]
                    start -= obj.n
                    stop -= obj.n
            return type(self)(members=members)


########## end of file ##########
