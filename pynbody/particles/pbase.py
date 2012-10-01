#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import sys
import copy
import hashlib
import numpy as np
from ..lib import gravity
from ..lib.utils.timing import decallmethods, timings
from ..lib.utils.memoize import cache


__all__ = ['Pbase']


def make_attrs(cls):
    def make_property(attr, doc):
        def fget(self): return self.data[attr]
        def fset(self, value): self.data[attr] = value
        def fdel(self): raise NotImplementedError()
        return property(fget, fset, fdel, doc)
    attrs = ((i[0], cls.__name__+'\'s '+i[2]) for i in cls.attrs)
    for (attr, doc) in attrs:
        setattr(cls, attr, make_property(attr, doc))
    return cls



class AbstractNbodyUtils(object):
    """

    """

    def __len__(self):
        return len(self.id)


    def __contains__(self, id):
        return id in self.id


    def copy(self):
        return copy.deepcopy(self)



class AbstractNbodyMethods(AbstractNbodyUtils):
    """
    This class holds common methods for particles in n-body systems.
    """
    common_attrs = [# name, dtype, doc
                    ('id', 'u8', 'index'),
                    ('mass', 'f8', 'mass'),
                    ('pos', '3f8', 'position'),
                    ('vel', '3f8', 'velocity'),
                    ('acc', '3f8', 'acceleration'),
                    ('phi', 'f8', 'potential'),
                    ('eps2', 'f8', 'squared softening'),
                    ('time', 'f8', 'current time'),
                    ('tstep', 'f8', 'time step'),
                    ('nstep', 'u8', 'step number'),
                   ]
    common_names = [_[0] for _ in common_attrs]
    common_dtype = [(_[0], _[1]) for _ in common_attrs]
    common_data0 = np.zeros(0, common_dtype)


    ### total mass and center-of-mass

    @property
    @cache
    def total_mass(self):
        """
        Total mass.
        """
        return float(self.mass.sum())

    @property
    @cache
    def rcom(self):
        """
        Position of the center-of-mass.
        """
        mtot = self.total_mass
        rcom = (self.mass * self.pos.T).sum(1)
        return (rcom / mtot)

    @property
    @cache
    def vcom(self):
        """
        Velocity of the center-of-mass.
        """
        mtot = self.total_mass
        vcom = (self.mass * self.vel.T).sum(1)
        return (vcom / mtot)

    def move_to_center(self):
        """
        Moves the center-of-mass to the origin of coordinates.
        """
        self.pos -= self.rcom
        self.vel -= self.vcom


    ### linear momentum

    @property
    @cache
    def lm(self):
        """
        Individual linear momentum.
        """
        return (self.mass * self.vel.T).T

    @property
    @cache
    def linear_momentum(self):
        """
        Total linear momentum.
        """
        return self.lm.sum(0)


    ### angular momentum

    @property
    @cache
    def am(self):
        """
        Individual angular momentum.
        """
        return (self.mass * np.cross(self.pos, self.vel).T).T

    @property
    @cache
    def angular_momentum(self):
        """
        Total angular momentum.
        """
        return self.am.sum(0)


    ### kinetic energy

    @property
    @cache
    def ke(self):
        """
        Individual kinetic energy.
        """
        return 0.5 * self.mass * (self.vel**2).sum(1)

    @property
    @cache
    def kinetic_energy(self):
        """
        Total kinetic energy.
        """
        return float(self.ke.sum())


    ### potential energy

    @property
    @cache
    def pe(self):
        """
        Individual potential energy.
        """
        phi = self.get_phi(self)
        return self.mass * phi

    @property
    @cache
    def potential_energy(self):
        """
        Total potential energy.
        """
        return 0.5 * float(self.pe.sum())


    ### virial energy

    @property
    @cache
    def ve(self):
        """
        Individual virial energy.
        """
        acc = self.get_acc(self)
        force = (self.mass * acc.T).T
        return self.pos * force

    @property
    @cache
    def virial_energy(self):
        """
        Total virial energy.
        """
        return float(self.ve.sum())



    ### gravity

    def get_tstep(self, objs, eta):
        """
        Get the individual time-steps due to other particles.
        """
        gravity.tstep.set_args(self, objs, eta/2)
        gravity.tstep.run()
        return gravity.tstep.get_result()

    def get_phi(self, objs):
        """
        Get the individual gravitational potential due to other particles.
        """
        gravity.phi.set_args(self, objs)
        gravity.phi.run()
        return gravity.phi.get_result()

    def get_acc(self, objs):
        """
        Get the individual gravitational acceleration due to other particles.
        """
        gravity.acc.set_args(self, objs)
        gravity.acc.run()
        return gravity.acc.get_result()

    def get_acc_jerk(self, objs):
        """
        Get the individual gravitational acceleration and jerk due to other particles.
        """
        gravity.acc_jerk.set_args(self, objs)
        gravity.acc_jerk.run()
        return gravity.acc_jerk.get_result()


    def update_tstep(self, objs, eta):
        """
        Update the individual time-steps due to other particles.
        """
        self.tstep = self.get_tstep(objs, eta)

    def update_phi(self, objs):
        """
        Update the individual gravitational potential due to other particles.
        """
        self.phi = self.get_phi(objs)

    def update_acc(self, objs):
        """
        Update the individual gravitational acceleration due to other particles.
        """
        self.acc = self.get_acc(objs)

    def update_acc_jerk(self, objs):
        """
        Update the individual gravitational acceleration and jerk due to other particles.
        """
        (self.acc, self.jerk) = self.get_acc_jerk(objs)


    ### miscellaneous methods

    def min_tstep(self):
        """
        Minimum absolute value of tstep.
        """
        return np.abs(self.tstep).min()

    def max_tstep(self):
        """
        Maximum absolute value of tstep.
        """
        return np.abs(self.tstep).max()




@decallmethods(timings)
class Pbase(AbstractNbodyMethods):
    """

    """
    attrs = None
    names = None
    dtype = None
    data0 = None

    def __init__(self, n=0, data=None):
        if data is None:
            if n: data = np.zeros(n, self.dtype)
            else: data = self.data0
        self.data = data
        self.n = len(self)


    def items(self):
        return [(type(self).__name__.lower(), self)]


    #
    # miscellaneous methods
    #

    def __str__(self):
        fmt = type(self).__name__+'(['
        if self.n:
            fmt += '\n'
            for obj in self:
                fmt += '{\n'
                for name in obj.data.dtype.names:
                    fmt += ' {0}: {1},\n'.format(name, getattr(obj, name).tolist()[0])
                fmt += '},\n'
        fmt += '])'
        return fmt


    def __repr__(self):
        fmt = type(self).__name__+'('
        if self.n: fmt += str(self.data)
        else: fmt += '[]'
        fmt += ')'
        return fmt


    def __hash__(self):
        return int(hashlib.md5(self.data).hexdigest(), 32) % sys.maxint


    def __getitem__(self, slc):
        if isinstance(slc, int):
            data = self.data[[slc]]
        if isinstance(slc, slice):
            data = self.data[slc]
        if isinstance(slc, list):
            data = self.data[slc]
        if isinstance(slc, np.ndarray):
            if slc.all():
                return self
            if slc.any():
                data = self.data[slc]
            else:
                data = None

        return type(self)(data=data)


    def append(self, obj):
        if obj.n:
            self.data = np.concatenate((self.data, obj.data))
            self.n = len(self)


    def remove(self, id):
        slc = np.where(self.id == id)
        self.data = np.delete(self.data, slc, 0)
        self.n = len(self)


    def insert(self, id, obj):
        index = np.where(self.id == id)[0]
        v = obj.data
        self.data = np.insert(self.data, index*np.ones(len(v)), v, 0)
        self.n = len(self)


    def pop(self, id=None):
        if id is None:
            index = -1
            id = self.id[-1]
        else:
            index = np.where(self.id == id)[0]
        obj = self[index]
        self.remove(id)
        return obj


    def get_state(self):
        return self.data


    def set_state(self, state):
        self.data = type(self)(len(state)).data
        self.data[:] = state
        self.n = len(self)


    def astype(self, cls):
        obj = cls()
        obj.set_state(self.get_state())
        return obj


########## end of file ##########
