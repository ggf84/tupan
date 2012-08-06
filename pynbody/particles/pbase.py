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


__all__ = ['Pbase']


def make_attrs(cls):
    def make_property(attr, doc):
        def fget(self): return getattr(self, attr)
        def fset(self, value): getattr(self, attr)[:] = value
        def fdel(self): raise NotImplementedError()
        return property(fget, fset, fdel, doc)
    attrs = ((i[0], cls.__name__+'\'s '+i[2]) for i in cls.attributes)
    for (attr, doc) in attrs:
        setattr(cls, attr, make_property(attr, doc))
    return cls


def empty_copy(obj, *args, **kwargs):
    return obj.__class__(*args, **kwargs)

new = empty_copy


class AbstractNbodyUtils(object):
    """

    """

    def copy(self):
        return copy.deepcopy(self)



class AbstractNbodyMethods(AbstractNbodyUtils):
    """
    This class holds common methods for particles in n-body systems.
    """
    common_attributes = [# name, dtype, doc
                         ('id', 'u8', 'index'),
                         ('mass', 'f8', 'mass'),
                         ('pos', '3f8', 'position'),
                         ('vel', '3f8', 'velocity'),
                         ('acc', '3f8', 'acceleration'),
                         ('phi', 'f8', 'potential'),
                         ('eps2', 'f8', 'softening'),
                         ('t_curr', 'f8', 'current time'),
                         ('dt_prev', 'f8', 'previous time-step'),
                         ('dt_next', 'f8', 'next time-step'),
                         ('nstep', 'u8', 'step number'),
                        ]

    common_names = [_[0] for _ in common_attributes]


    ### total mass and center-of-mass

    def get_total_mass(self):
        """
        Get the total mass.
        """
        return float(self.mass.sum())

    def get_center_of_mass_position(self):
        """
        Get the center-of-mass position.
        """
        mtot = self.get_total_mass()
        rcom = (self.mass * self.pos.T).sum(1)
        return (rcom / mtot)

    def get_center_of_mass_velocity(self):
        """
        Get the center-of-mass velocity.
        """
        mtot = self.get_total_mass()
        vcom = (self.mass * self.vel.T).sum(1)
        return (vcom / mtot)

    def correct_center_of_mass(self):
        """
        Correct the center-of-mass to origin of coordinates.
        """
        self.pos -= self.get_center_of_mass_position()
        self.vel -= self.get_center_of_mass_velocity()


    ### linear momentum

    def get_total_linear_momentum(self):
        """
        Get the total linear momentum.
        """
        return (self.mass * self.vel.T).sum(1)


    ### angular momentum

    def get_total_angular_momentum(self):
        """
        Get the total angular momentum.
        """
        return (self.mass * np.cross(self.pos, self.vel).T).sum(1)


    ### kinetic energy

    def get_total_kinetic_energy(self):
        """
        Get the total kinetic energy.
        """
        return float((0.5 * self.mass * (self.vel**2).sum(1)).sum())


    ### potential energy

    def get_total_potential_energy(self):
        """
        Get the total potential energy.
        """
        return 0.5 * float((self.mass * self.phi).sum())


    ### gravity

    def update_tstep(self, objs, eta):
        """
        Update the individual time-steps due to other particles.
        """
        gravity.tstep.set_args(self, objs, eta/2)
        gravity.tstep.run()
        self.dt_next = gravity.tstep.get_result()

    def update_phi(self, objs):
        """
        Update the individual gravitational potential due to other particles.
        """
        gravity.phi.set_args(self, objs)
        gravity.phi.run()
        self.phi = gravity.phi.get_result()

    def update_acc(self, objs):
        """
        Update the individual gravitational acceleration due to other particles.
        """
        gravity.acc.set_args(self, objs)
        gravity.acc.run()
        self.acc = gravity.acc.get_result()

    def update_acc_jerk(self, objs):
        """
        Update the individual gravitational acceleration and jerk due to other particles.
        """
        gravity.acc_jerk.set_args(self, objs)
        gravity.acc_jerk.run()
        (self.acc, self.jerk) = gravity.acc_jerk.get_result()


    ### miscellaneous methods

    def min_dt_prev(self):
        """
        Minimum absolute value of dt_prev.
        """
        return np.abs(self.dt_prev).min()

    def max_dt_prev(self):
        """
        Maximum absolute value of dt_prev.
        """
        return np.abs(self.dt_prev).max()


    def min_dt_next(self):
        """
        Minimum absolute value of dt_next.
        """
        return np.abs(self.dt_next).min()

    def max_dt_next(self):
        """
        Maximum absolute value of dt_nex.
        """
        return np.abs(self.dt_next).max()



@decallmethods(timings)
class Pbase(AbstractNbodyMethods):
    """

    """
    attributes = None
    names = None
    dtype = None
    data0 = None

    def __init__(self, n=0, data=None):
        if data is None:
            if n: data = {attr: np.zeros(n, dtype) for (attr, dtype) in self.dtype}
            else: data = {attr: self.data0[attr] for attr in self.names}
        self.__dict__ = data


    @property
    def items(self):
        return [(self.__class__.__name__.lower(), self)]


    #
    # miscellaneous methods
    #

    def __str__(self):
        fmt = self.__class__.__name__+'(['
        if self.n:
            fmt += '\n'
            for obj in self:
                fmt += '{\n'
                for name in self.names:
                    fmt += ' {0}: {1},\n'.format(name, vars(obj)[name].tolist())
                fmt += '},\n'
        fmt += '])'
        return fmt


    def __repr__(self):
        return str(vars(self))


    def __hash__(self):
        return int(hashlib.md5(self.id).hexdigest(), 32) % sys.maxint


    def __len__(self):
        return len(self.id)
    n = property(__len__)


    def __getitem__(self, slc):
        data = {k: v[[slc]] for (k, v) in vars(self).items()}
        return new(self, data=data)


    def append(self, obj):
        for (k, v) in vars(obj).items():
            setattr(self, k, np.concatenate((getattr(self, k), v)))


    def remove(self, id):
        slc = np.where(self.id == id)
        for k in vars(self).keys():
            setattr(self, k, np.delete(getattr(self, k), slc, 0))


    def insert(self, id, obj):
        index = np.where(self.id == id)[0]
        for k in vars(self).keys():
            v = getattr(obj, k)
            setattr(self, k, np.insert(getattr(self, k), index*np.ones(len(v)), v, 0))


    def pop(self, id=None):
        if id is None:
            index = -1
            id = self.id[-1]
        else:
            index = np.where(self.id == id)[0]
        obj = self[index]
        self.remove(id)
        return obj


    def get_state(self):    # XXX
        data = np.zeros(self.n, self.dtype)
        for (k, v) in vars(self).items():
            data[k][:] = v
        return data[self.common_names]


    def set_state(self, state):
        self.__dict__ = new(self, len(state)).__dict__
        for name in state.dtype.names:
            if name in self.names:
                getattr(self, name)[:] = state[name]


    def astype(self, cls):
        obj = cls()
        obj.set_state(self.get_state())
        return obj


    def select(self, function):
        return self[function(self)]


    #
    # common methods
    #

    ### linear momentum

    def get_individual_linear_momentum(self):
        """
        Get the individual linear momentum.
        """
        return (self.mass * self.vel.T).T


    ### angular momentum

    def get_individual_angular_momentum(self):
        """
        Get the individual angular momentum.
        """
        return (self.mass * np.cross(self.pos, self.vel).T).T


    ### kinetic energy

    def get_individual_kinetic_energy(self):
        """
        Get the individual kinetic energy.
        """
        return 0.5 * self.mass * (self.vel**2).sum(1)


    ### potential energy

    def get_individual_potential_energy(self):
        """
        Get the individual potential energy.
        """
        return self.mass * self.phi


########## end of file ##########
