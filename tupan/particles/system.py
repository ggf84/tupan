# -*- coding: utf-8 -*-
#

"""
TODO.
"""

from __future__ import print_function
import copy
import numpy as np
from .body import Body
from .sph import Sph
from .star import Star
from .blackhole import Blackhole
from .base import MetaParticle, AbstractNbodyMethods
from ..lib.utils import with_metaclass


class Members(dict):
    def __init__(self, **members):
        super(Members, self).__init__(**members)
        self.__dict__ = self


class ParticleSystem(with_metaclass(MetaParticle, AbstractNbodyMethods)):
    """
    This class holds the particle types in the simulation.
    """
    name = None

    def __init__(self, nbody=0, nstar=0, nbh=0, nsph=0):
        """
        Initializer.
        """
        members = {cls.name: cls(n) for (n, cls) in [(nbody, Body),
                                                     (nstar, Star),
                                                     (nbh, Blackhole),
                                                     (nsph, Sph)] if n}
        self.set_members(**members)
        if self.n:
            self.reset_pid()

    def reset_pid(self):
        self.pid[...] = range(self.n)

    def set_members(self, **members):
        self.members = Members(**members)
        self.n = len(self)
        for member in self.members.values():
            self.attr_names.update(**member.attr_names)

    def update_members(self, **members):
        vars(self).clear()
        self.set_members(**members)

    @classmethod
    def from_members(cls, **members):
        obj = cls.__new__(cls)
        obj.set_members(**members)
        return obj

    def register_attribute(self, name, shape, sctype, doc=''):
        for member in self.members.values():
            member.register_attribute(name, shape, sctype, doc)

        if name not in self.attr_names:
            self.attr_names[name] = (shape, sctype, doc)

    #
    # miscellaneous methods
    #

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return repr(vars(self))

    def __str__(self):
        fmt = self.name + '(['
        if self.n:
            for member in self.members.values():
                fmt += '\n\t{0},'.format('\n\t'.join(str(member).split('\n')))
            fmt += '\n'
        fmt += '])'
        return fmt

    def __len__(self):
        return sum(len(member) for member in self.members.values())

    def __add__(self, other):
        if other.n:
            members = self.members
            for name, member in other.members.items():
                try:
                    members[name] += member
                except KeyError:
                    members[name] = member.copy()
            self.update_members(**members)
        return self

    __radd__ = __add__

    def split_by(self, mask_function):
        d_a, d_b = {}, {}
        for name, member in self.members.items():
            d_a[name], d_b[name] = member.split_by(mask_function)
        return (self.from_members(**d_a),
                self.from_members(**d_b))

    def __getitem__(self, index):
        if isinstance(index, np.ndarray):
            ns, nf = 0, 0
            members = {}
            for name, member in self.members.items():
                nf += member.n
                members[name] = member[index[ns:nf]]
                ns += member.n
            return self.from_members(**members)

        if isinstance(index, int):
            if abs(index) > self.n-1:
                msg = 'index {0} out of bounds 0<=index<{1}'
                raise IndexError(msg.format(index, self.n))
            if index < 0:
                index = self.n + index
            members = {}
            for name, member in self.members.items():
                if 0 <= index < member.n:
                    members[name] = member[index]
                index -= member.n
            return self.from_members(**members)

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
            for name, member in self.members.items():
                if stop >= 0 and start < member.n:
                    members[name] = member[start-member.n:stop]
                start -= member.n
                stop -= member.n
            return self.from_members(**members)

    def __setitem__(self, index, value):
        if isinstance(index, np.ndarray):
            ns, nf = 0, 0
            for name, member in self.members.items():
                nf += member.n
                if name in value.members:
                    member[index[ns:nf]] = value.members[name]
                ns += member.n
            return

        if isinstance(index, int):
            if abs(index) > self.n-1:
                msg = 'index {0} out of bounds 0<=index<{1}'
                raise IndexError(msg.format(index, self.n))
            if index < 0:
                index = self.n + index
            for name, member in self.members.items():
                if 0 <= index < member.n:
                    if name in value.members:
                        member[index] = value.members[name]
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
            for name, member in self.members.items():
                if stop >= 0 and start < member.n:
                    if name in value.members:
                        member[start-member.n:stop] = value.members[name]
                start -= member.n
                stop -= member.n
            return

    def __getattr__(self, name):
        if name not in self.attr_names:
            raise AttributeError(name)
        members = self.members
        if len(members) == 1:
            member = next(iter(members.values()))
            value = getattr(member, name)
            setattr(self, name, value)
            return value
        arrays = [getattr(member, name) for member in members.values()
                  if name in member.attr_names]
        value = np.concatenate(arrays, -1)  # along last dimension
        ns, nf = 0, 0
        for member in members.values():
            if name in member.attr_names:
                nf += member.n
                setattr(member, name, value[..., ns:nf])
                ns += member.n
        setattr(self, name, value)
        return value


# -- End of File --
