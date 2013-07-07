# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import copy
import numpy as np
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
    def __init__(self, nbodies=0, nstars=0, nbhs=0, nsphs=0):
        """
        Initializer
        """
        members = {cls.__name__.lower(): cls(n)
                   for n, cls in [(nbodies, Bodies),
                                  (nstars, Stars),
                                  (nbhs, Blackholes),
                                  (nsphs, Sphs)] if n}
        self.members = members
        self.n = len(self)
        if self.n:
            self._rebind_attrs()

    def _rebind_attrs(self):
        for k, v in self.members.items():
            setattr(self, k, v)
        objs = self.members.values()
        attrlist = {attr for obj in objs for (attr, _, _) in obj.attrs}
        for attr in attrlist:
            seq = [getattr(obj, attr) for obj in objs if hasattr(obj, attr)]
            ary = np.concatenate(seq)
            setattr(self, attr, ary)
            ns = 0
            nf = 0
            for obj in objs:
                if hasattr(obj, attr):
                    nf += obj.n
                    setattr(obj, attr, ary[ns:nf])
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
        if self.n:
            self._rebind_attrs()

    def __getitem__(self, slc):
        if isinstance(slc, (list, np.ndarray)):
            if all(slc):
                return self
            if any(slc):
                ns = 0
                nf = 0
                subset = type(self)()
                for obj in self.members.values():
                    nf += obj.n
                    subset.append(obj[slc[ns:nf]])
                    ns += obj.n
                return subset
            return type(self)()

        if isinstance(slc, int):
            subset = type(self)()
            if abs(slc) > self.n-1:
                raise IndexError(
                    "index {0} out of bounds 0<=index<{1}".format(slc, self.n))
            if slc < 0:
                slc = self.n + slc
            n = 0
            for obj in self.members.values():
                i = slc - n
                if 0 <= i < obj.n:
                    subset.append(obj[i])
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
