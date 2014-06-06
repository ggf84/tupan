# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import numpy as np
from .body import Bodies
from .sph import Sphs
from .star import Stars
from .blackhole import Blackholes
from .base import AbstractNbodyMethods
from ..lib.utils.timing import timings, bind_all


__all__ = ['ParticleSystem']


@bind_all(timings)
class ParticleSystem(AbstractNbodyMethods):
    """
    This class holds the particle types in the simulation.
    """
    def __init__(self, nbodies=0, nstars=0, nbhs=0, nsphs=0):
        """
        Initializer.
        """
        self.members = {cls.__name__.lower(): cls(n)
                        for (n, cls) in [(nbodies, Bodies),
                                         (nstars, Stars),
                                         (nbhs, Blackholes),
                                         (nsphs, Sphs)] if n}
        self.n = len(self)
        if self.n:
            self.id[...] = range(self.n)

    def update_members(self, members):
        d = self.__dict__
        m = d.get('members', {})
        m.update(members)
        d.clear()
        self.members = m
        self.n = len(self)

    @classmethod
    def from_members(cls, members):
        obj = cls.__new__(cls)
        obj.update_members(members)
        return obj

    def register_attribute(self, name, sctype, doc=''):
        for member in self.members.values():
            member.register_attribute(name, sctype, doc)

        super(ParticleSystem, self).register_attribute(name, sctype, doc)

    #
    # miscellaneous methods
    #

    def __str__(self):
        fmt = type(self).__name__ + '(['
        if self.n:
            for member in self.members.values():
                fmt += '\n\t{0},'.format('\n\t'.join(str(member).split('\n')))
            fmt += '\n'
        fmt += '])'
        return fmt

    def __len__(self):
        return sum(len(member) for member in self.members.values())

    def append(self, obj):
        if obj.n:
            members = self.members
            for (k, v) in obj.members.items():
                try:
                    members[k].append(v)
                except KeyError:
                    members[k] = v.copy()
            self.update_members(members)

    def __getitem__(self, index):
        if isinstance(index, np.ndarray):
            ns = 0
            nf = 0
            members = {}
            for (key, obj) in self.members.items():
                nf += obj.n
                members[key] = obj[index[ns:nf]]
                ns += obj.n
            return self.from_members(members)

        if isinstance(index, int):
            if abs(index) > self.n-1:
                msg = 'index {0} out of bounds 0<=index<{1}'
                raise IndexError(msg.format(index, self.n))
            if index < 0:
                index = self.n + index
            members = {}
            for (key, obj) in self.members.items():
                if 0 <= index < obj.n:
                    members[key] = obj[index]
                index -= obj.n
            return self.from_members(members)

        if isinstance(index, slice):
            start = index.start
            stop = index.stop
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
                if stop >= 0 and start < obj.n:
                    members[key] = obj[start-obj.n:stop]
                start -= obj.n
                stop -= obj.n
            return self.from_members(members)

    def __setitem__(self, index, values):
        if isinstance(index, np.ndarray):
            ns = 0
            nf = 0
            for (key, obj) in self.members.items():
                nf += obj.n
                obj[index[ns:nf]] = values.members[key]
                ns += obj.n
            return

        if isinstance(index, int):
            if abs(index) > self.n-1:
                msg = 'index {0} out of bounds 0<=index<{1}'
                raise IndexError(msg.format(index, self.n))
            if index < 0:
                index = self.n + index
            for (key, obj) in self.members.items():
                if 0 <= index < obj.n:
                    obj[index] = values.members[key]
                index -= obj.n
            return

        if isinstance(index, slice):
            start = index.start
            stop = index.stop
            if start is None:
                start = 0
            if stop is None:
                stop = self.n
            if start < 0:
                start = self.n + start
            if stop < 0:
                stop = self.n + stop
            for (key, obj) in self.members.items():
                if stop >= 0 and start < obj.n:
                    obj[start-obj.n:stop] = values.members[key]
                start -= obj.n
                stop -= obj.n
            return

    def _init_lazyproperty(self, lazyprop):
        name = lazyprop.name
        members = self.members.values()
        if len(members) == 1:
            value = getattr(next(iter(members)), name)
            setattr(self, name, value)
            return value
        arrays = [getattr(member, name) for member in members]
        value = np.concatenate(arrays)
        ns = 0
        nf = 0
        for member in members:
            nf += member.n
            setattr(member, name, value[ns:nf])
            ns += member.n
        setattr(self, name, value)
        return value


# -- End of File --
