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
from .base import MetaParticle, AbstractNbodyMethods
from ..lib.utils import with_metaclass
from ..lib.utils.timing import timings, bind_all


__all__ = ['ParticleSystem']


@bind_all(timings)
class Members(list):
    def __init__(self, members):
        super(Members, self).__init__(members)
        for member in self:
            setattr(self, member.name, member)

    def __contains__(self, key):
        return key in vars(self)


@bind_all(timings)
class ParticleSystem(with_metaclass(MetaParticle, AbstractNbodyMethods)):
    """
    This class holds the particle types in the simulation.
    """
    name = None

    def __init__(self, nbodies=0, nstars=0, nbhs=0, nsphs=0):
        """
        Initializer.
        """
        members = [cls(n) for (n, cls) in [(nbodies, Bodies),
                                           (nstars, Stars),
                                           (nbhs, Blackholes),
                                           (nsphs, Sphs)] if n]
        self.set_members(members)
        if self.n:
            self.id[...] = range(self.n)

    def set_members(self, members):
        self.members = Members(members)
        self.n = len(self)

    def update_members(self, members):
        vars(self).clear()
        self.set_members(members)

    @classmethod
    def from_members(cls, members):
        obj = cls.__new__(cls)
        obj.set_members(members)
        return obj

    def register_attribute(self, name, shape, sctype, doc=''):
        for member in self.members:
            member.register_attribute(name, shape, sctype, doc)

        if name not in self.attr_names:
            self.attr_names.append(name)
            self.attr_descrs[name] = (shape, sctype, doc)

    #
    # miscellaneous methods
    #

    def __delattr__(self, name):
        for member in self.members:
            if hasattr(member, name):
                delattr(member, name)
        super(ParticleSystem, self).__delattr__(name)

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return repr(vars(self))

    def __str__(self):
        fmt = self.name + '(['
        if self.n:
            for member in self.members:
                fmt += '\n\t{0},'.format('\n\t'.join(str(member).split('\n')))
            fmt += '\n'
        fmt += '])'
        return fmt

    def __len__(self):
        return sum(len(member) for member in self.members)

    def append(self, obj):
        if obj.n:
            members = self.members
            for (i, member) in enumerate(obj.members):
                try:
                    members[i].append(member)
                except IndexError:
                    members.append(member.copy())
            self.update_members(members)

    def __getitem__(self, index):
        if isinstance(index, np.ndarray):
            ns = 0
            nf = 0
            members = []
            for member in self.members:
                nf += member.n
                members.append(member[index[ns:nf]])
                ns += member.n
            return self.from_members(members)

        if isinstance(index, int):
            if abs(index) > self.n-1:
                msg = 'index {0} out of bounds 0<=index<{1}'
                raise IndexError(msg.format(index, self.n))
            if index < 0:
                index = self.n + index
            members = []
            for member in self.members:
                if 0 <= index < member.n:
                    members.append(member[index])
                index -= member.n
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
            members = []
            for member in self.members:
                if stop >= 0 and start < member.n:
                    members.append(member[start-member.n:stop])
                start -= member.n
                stop -= member.n
            return self.from_members(members)

    def __setitem__(self, index, values):
        if isinstance(index, np.ndarray):
            ns = 0
            nf = 0
            for (key, member) in enumerate(self.members):
                nf += member.n
                member[index[ns:nf]] = values.members[key]
                ns += member.n
            return

        if isinstance(index, int):
            if abs(index) > self.n-1:
                msg = 'index {0} out of bounds 0<=index<{1}'
                raise IndexError(msg.format(index, self.n))
            if index < 0:
                index = self.n + index
            for (key, member) in enumerate(self.members):
                if 0 <= index < member.n:
                    member[index] = values.members[key]
                index -= member.n
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
            for (key, member) in enumerate(self.members):
                if stop >= 0 and start < member.n:
                    member[start-member.n:stop] = values.members[key]
                start -= member.n
                stop -= member.n
            return

    def __getattr__(self, name):
        if name not in self.attr_names:
            raise AttributeError(name)
        shape, _, _ = self.attr_descrs[name]
        members = self.members
        if len(members) == 1:
            member = next(iter(members))
            value = getattr(member, name)
            value = np.array(value, copy=False, order='C', ndmin=1)
            setattr(member, name, value)
            setattr(self, name, value)
            return value
        concat = np.concatenate
        arrays = [getattr(member, name) for member in members]
        value = concat(arrays, 1) if shape else concat(arrays)
        value = np.array(value, copy=False, order='C', ndmin=1)
        ns = 0
        nf = 0
        for member in members:
            nf += member.n
            setattr(member, name, value[..., ns:nf])
            ns += member.n
        setattr(self, name, value)
        return value


# -- End of File --
