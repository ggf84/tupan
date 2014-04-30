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
from ..lib.utils.timing import timings, bind_all


__all__ = ["ParticleSystem"]


@bind_all(timings)
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
        for member in self.members.values():
            for (attr, _) in member.attributes:
                attrs[attr].append(getattr(member, attr))

        for (attr, arrays) in attrs.items():
            value = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
            setattr(self, attr, value)

        ns = 0
        nf = 0
        for member in self.members.values():
            nf += member.n
            for (attr, _) in member.attributes:
                setattr(member, attr, getattr(self, attr)[ns:nf])
            ns += member.n

    def register_attribute(self, attr, sctype, doc=''):
        for member in self.members.values():
            member.register_attribute(attr, sctype, doc)

        super(ParticleSystem, self).register_attribute(attr, sctype, doc)

    #
    # miscellaneous methods
    #

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        fmt = type(self).__name__+"(["
        if self.n:
            for member in self.members.values():
                fmt += "\n\t{0},".format('\n\t'.join(str(member).split('\n')))
            fmt += "\n"
        fmt += "])"
        return fmt

    def __len__(self):
        return sum(len(member) for member in self.members.values())

    def copy(self):
        return copy.deepcopy(self)

    def append(self, obj):
        if obj.n:
            try:
                members = obj.members
            except AttributeError:
                members = dict([(type(obj).__name__.lower(), obj)])
            for (k, v) in members.items():
                try:
                    self.members[k].append(v)
                except KeyError:
                    self.members[k] = v.copy()
            self.n = len(self)
            for attr in list(self.__dict__):
                if attr not in ('n', 'members'):
                    delattr(self, attr)

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
                obj[slc[ns:nf]] = values.members[key]
                ns += obj.n
            if self.n:
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
                    obj[i] = values.members[key]
                n += obj.n
            if self.n:
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
                            obj[start-obj.n:stop] = values.members[key]
                    start -= obj.n
                    stop -= obj.n
            if self.n:
                self._rebind_attrs()
            return


# -- End of File --
