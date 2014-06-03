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


def particle_system_property(name, sctype,
                             doc=None, can_get=True,
                             can_set=True, can_del=True):
    storage_name = '_' + name

    def fget(self):
        value = getattr(self, storage_name, None)
        if value is None:
            arrays = [getattr(member, name)
                      for member in self.members.values()]
            value = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
            ns = 0
            nf = 0
            for member in self.members.values():
                nf += member.n
                setattr(member, name, value[ns:nf])
                ns += member.n
            setattr(self, storage_name, value)
        return value

    def fset(self, value):
        setattr(self, storage_name, value)

    def fdel(self):
        if hasattr(self, storage_name):
            delattr(self, storage_name)
        for member in self.members.values():
            delattr(member, name)

    fget.__name__ = name
    fset.__name__ = name
    fdel.__name__ = name
    return property(fget if can_get else None,
                    fset if can_set else None,
                    fdel if can_del else None,
                    doc)


@bind_all(timings)
class ParticleSystem(AbstractNbodyMethods):
    """
    This class holds the particle types in the simulation.
    """
    def __init__(self, nbodies=0, nstars=0, nbhs=0, nsphs=0):
        """
        Initializer.
        """
        members = {cls.__name__.lower(): cls(n)
                   for (n, cls) in [(nbodies, Bodies),
                                    (nstars, Stars),
                                    (nbhs, Blackholes),
                                    (nsphs, Sphs)] if n}
        self.set_members(members)
        if self.n:
            self.id[...] = range(self.n)

    def set_members(self, members):
        self.members = members
        self.n = len(self)

    @classmethod
    def from_members(cls, members):
        obj = cls.__new__(cls)
        obj.set_members(members)
        return obj

    #
    # miscellaneous methods
    #

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        fmt = type(self).__name__+'(['
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
            ns = 0
            nf = 0
            members = {}
            for (key, obj) in self.members.items():
                nf += obj.n
                members[key] = obj[slc[ns:nf]]
                ns += obj.n
            return type(self).from_members(members)

        if isinstance(slc, int):
            if abs(slc) > self.n-1:
                raise IndexError(
                    'index {0} out of bounds 0<=index<{1}'.format(slc, self.n))
            if slc < 0:
                slc = self.n + slc
            i = slc
            members = {}
            for (key, obj) in self.members.items():
                if 0 <= i < obj.n:
                    members[key] = obj[i]
                i -= obj.n
            return type(self).from_members(members)

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
                if stop >= 0 and start < obj.n:
                    members[key] = obj[start-obj.n:stop]
                start -= obj.n
                stop -= obj.n
            return type(self).from_members(members)

    def __setitem__(self, slc, values):
        if isinstance(slc, np.ndarray):
            ns = 0
            nf = 0
            for (key, obj) in self.members.items():
                nf += obj.n
                obj[slc[ns:nf]] = values.members[key]
                ns += obj.n
            return

        if isinstance(slc, int):
            if abs(slc) > self.n-1:
                raise IndexError(
                    'index {0} out of bounds 0<=index<{1}'.format(slc, self.n))
            if slc < 0:
                slc = self.n + slc
            i = slc
            for (key, obj) in self.members.items():
                if 0 <= i < obj.n:
                    obj[i] = values.members[key]
                i -= obj.n
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
                if stop >= 0 and start < obj.n:
                    obj[start-obj.n:stop] = values.members[key]
                start -= obj.n
                stop -= obj.n
            return

    @classmethod
    def register_property(cls, name, sctype, doc=''):
        setattr(cls, name, particle_system_property(name, sctype, doc))

    def register_attribute(self, name, sctype, doc=''):
        for member in self.members.values():
            member.register_attribute(name, sctype, doc)
        type(self).register_property(name, sctype, doc)


# -- End of File --
