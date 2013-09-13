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
        Initializer.
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
            setattr(self, key, obj)
            for attr in obj.__dict__:
                attrs[attr].append(getattr(obj, attr))

        for (attr, seq) in attrs.items():
            ary = np.concatenate(seq) if len(seq) > 1 else seq[0]
            setattr(self, attr, ary)

        if len(self.members) > 1:
            ns = 0
            nf = 0
            for obj in self.members.values():
                nf += obj.n
                for attr in obj.__dict__:
                    setattr(obj, attr, getattr(self, attr)[ns:nf])
                ns += obj.n

    def register_auxiliary_attribute(self, attr, dtype):
        if attr in self.__dict__:
            raise ValueError("'{0}' is already a registered "
                             "attribute.".format(attr))
        setattr(self, attr, np.zeros(self.n, dtype))
        ns = 0
        nf = 0
        for obj in self.members.values():
            nf += obj.n
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
                items = obj.members.items()
            except:
                items = [(type(obj).__name__.lower(), obj)]
            for (k, v) in items:
                try:
                    self.members[k].append(v)
                except:
                    self.members[k] = v.copy()
            self.n = len(self)
            self._rebind_attrs()

    def __getitem__(self, slc):
        if isinstance(slc, np.ndarray):
            members = {}
            ns = 0
            nf = 0
            for (key, obj) in self.members.items():
                nf += obj.n
                members[key] = obj[slc[ns:nf]]
                ns += obj.n
            return type(self)(members=members)

        if isinstance(slc, int):
            members = {}
            if abs(slc) > self.n-1:
                raise IndexError(
                    "index {0} out of bounds 0<=index<{1}".format(slc, self.n))
            if slc < 0:
                slc = self.n + slc
            n = 0
            for (key, obj) in self.members.items():
                i = slc - n
                if 0 <= i < obj.n:
                    members[key] = obj[i]
                n += obj.n
            return type(self)(members=members)

        if isinstance(slc, slice):
            members = {}
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
                            members[key] = obj[start-obj.n:stop]
                    start -= obj.n
                    stop -= obj.n
            return type(self)(members=members)

    def __setitem__(self, slc, values):
        if isinstance(slc, np.ndarray):
            ns = 0
            nf = 0
            for (key, obj) in self.members.items():
                nf += obj.n
                obj[slc[ns:nf]] = getattr(values, key)
                ns += obj.n
            self._rebind_attrs()
            return

        if isinstance(slc, int):
            if abs(slc) > self.n-1:
                raise IndexError(
                    "index {0} out of bounds 0<=index<{1}".format(slc, self.n))
            if slc < 0:
                slc = self.n + slc
            n = 0
            for (key, obj) in self.members.items():
                i = slc - n
                if 0 <= i < obj.n:
                    obj[i] = getattr(values, key)
                n += obj.n
            self._rebind_attrs()
            return

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
            for (key, obj) in self.members.items():
                if obj.n:
                    if stop >= 0:
                        if start < obj.n:
                            obj[start-obj.n:stop] = getattr(values, key)
                    start -= obj.n
                    stop -= obj.n
            self._rebind_attrs()
            return


########## end of file ##########
