#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import copy
import numpy as np
from ..lib.gravity import gravitation
from ..lib.utils.timing import decallmethods, timings


__all__ = ['Pbase']


@decallmethods(timings)
class Pbase(object):
    """

    """
    attrs = None
    dtype = None
    zero = None

    def __init__(self, n):
        self.data = np.zeros(n, self.dtype) if n else self.zero


    #
    # miscellaneous methods
    #

    def __repr__(self):
        fmt = self.__class__.__name__+'['
        if self:
            fmt += '\n'
            for item in self:
                if item:
                    fmt += '(\n'
                    for name in self.data.dtype.names:
                        fmt += '{0} = {1},\n'.format(name, getattr(item, name).tolist())
                    fmt += '),\n'
        fmt += ']'
        return fmt

    @property
    def n(self):
        return len(self.data)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        obj = copy.copy(self)
        obj.data = self.data[index]
        return obj

    def copy(self):
        return copy.deepcopy(self)

    def append(self, obj):
        self.data = np.append(self.data, obj.data)

    def remove(self, key):
        index = np.where(self.key == key)
        self.data = np.delete(self.data, index)

    def insert(self, index, obj):
        self.data = np.insert(self.data, index, obj.data)

    def pop(self, key=None):
        if key:
            index = np.where(self.key == key)
        else:
            index = -1
        obj = self[index]
        self.data = np.delete(self.data, index)
        return obj

    def get_state(self):
        return self.data[self.attrs]

    def set_state(self, state):
        self.data = np.zeros(len(state), dtype=self.dtype)
        for name in state.dtype.names:
            if name in self.data.dtype.names:
                self.data[name] = state[name]

    def astype(self, cls):
        obj = cls()
        obj.set_state(self.get_state())
        return obj

    def stack_fields(self, attrs):
        def atleast_2d(attr):
            arr = self.data[attr]
            return arr.reshape(-1,1) if arr.ndim < 2 else arr
        return np.concatenate([atleast_2d(attr) for attr in attrs], axis=1).squeeze()


    #
    # common attributes
    #

    ### id

    @property
    def id(self):
        return self.data['id']

    @id.setter
    def id(self, values):
        self.data['id'] = values

    @id.deleter
    def id(self):
        raise NotImplementedError()


    ### mass

    @property
    def mass(self):
        return self.data['mass']

    @mass.setter
    def mass(self, values):
        self.data['mass'] = values

    @mass.deleter
    def mass(self):
        raise NotImplementedError()


    ### pos

    @property
    def pos(self):
        return self.data['pos']

    @pos.setter
    def pos(self, values):
        self.data['pos'] = values

    @pos.deleter
    def pos(self):
        raise NotImplementedError()


    ### vel

    @property
    def vel(self):
        return self.data['vel']

    @vel.setter
    def vel(self, values):
        self.data['vel'] = values

    @vel.deleter
    def vel(self):
        raise NotImplementedError()


    ### acc

    @property
    def acc(self):
        return self.data['acc']

    @acc.setter
    def acc(self, values):
        self.data['acc'] = values

    @acc.deleter
    def acc(self):
        raise NotImplementedError()


    ### phi

    @property
    def phi(self):
        return self.data['phi']

    @phi.setter
    def phi(self, values):
        self.data['phi'] = values

    @phi.deleter
    def phi(self):
        raise NotImplementedError()


    ### eps2

    @property
    def eps2(self):
        return self.data['eps2']

    @eps2.setter
    def eps2(self, values):
        self.data['eps2'] = values

    @eps2.deleter
    def eps2(self):
        raise NotImplementedError()


    ### t_curr

    @property
    def t_curr(self):
        return self.data['t_curr']

    @t_curr.setter
    def t_curr(self, values):
        self.data['t_curr'] = values

    @t_curr.deleter
    def t_curr(self):
        raise NotImplementedError()


    ### dt_prev

    @property
    def dt_prev(self):
        return self.data['dt_prev']

    @dt_prev.setter
    def dt_prev(self, values):
        self.data['dt_prev'] = values

    @dt_prev.deleter
    def dt_prev(self):
        raise NotImplementedError()


    ### dt_next

    @property
    def dt_next(self):
        return self.data['dt_next']

    @dt_next.setter
    def dt_next(self, values):
        self.data['dt_next'] = values

    @dt_next.deleter
    def dt_next(self):
        raise NotImplementedError()


    #
    # common methods
    #

    ### total mass and center-of-mass

    def get_total_mass(self):
        """
        Get the total mass.
        """
        return float(np.sum(self.mass))

    def get_center_of_mass_position(self):
        """
        Get the center-of-mass position.
        """
        mtot = self.get_total_mass()
        return (self.mass * self.pos.T).sum(1) / mtot

    def get_center_of_mass_velocity(self):
        """
        Get the center-of-mass velocity.
        """
        mtot = self.get_total_mass()
        return (self.mass * self.vel.T).sum(1) / mtot

    def correct_center_of_mass(self):
        """
        Correct the center-of-mass to origin of coordinates.
        """
        self.pos -= self.get_center_of_mass_position()
        self.vel -= self.get_center_of_mass_velocity()


    ### linear momentum

    def get_individual_linear_momentum(self):
        """
        Get the individual linear momentum.
        """
        return (self.mass * self.vel.T).T

    def get_total_linear_momentum(self):
        """
        Get the total linear momentum.
        """
        return self.get_individual_linear_momentum().sum(0)


    ### angular momentum

    def get_individual_angular_momentum(self):
        """
        Get the individual angular momentum.
        """
        return (self.mass * np.cross(self.pos, self.vel).T).T

    def get_total_angular_momentum(self):
        """
        Get the total angular momentum.
        """
        return self.get_individual_angular_momentum().sum(0)


    ### kinetic energy

    def get_individual_kinetic_energy(self):
        """
        Get the individual kinetic energy.
        """
        return 0.5 * self.mass * (self.vel**2).sum(1)

    def get_total_kinetic_energy(self):
        """
        Get the total kinetic energy.
        """
        return float(np.sum(self.get_individual_kinetic_energy()))


    ### potential energy

    def get_individual_potential_energy(self):
        """
        Get the individual potential energy.
        """
        return self.mass * self.phi

    def get_total_potential_energy(self):
        """
        Get the total potential energy.
        """
        return 0.5 * float(np.sum(self.get_individual_potential_energy()))


    ### gravity

    def update_phi(self, jobj):
        nj = jobj.n
        jposmass = jobj.stack_fields(('pos', 'mass'))
        jeps2 = jobj.eps2
        self.phi = self.get_phi(nj, jposmass, jeps2)

    def update_acc(self, jobj):
        nj = jobj.n
        jposmass = jobj.stack_fields(('pos', 'mass'))
        jeps2 = jobj.eps2
        self.acc = self.get_acc(nj, jposmass, jeps2)

    def update_timestep(self, jobj, eta):
        nj = jobj.n
        jposmass = jobj.stack_fields(('pos', 'mass'))
        jveleps2 = jobj.stack_fields(('vel', 'eps2'))
        self.dt_next = self.get_tstep(nj, jposmass, jveleps2, eta)

    def update_acc_and_timestep(self, jobj, eta):   # XXX: deprecated!
        ret = iobj.get_acctstep(jobj, eta)
        self.acc = ret[0]
        self.dt_next = ret[1]


    def get_phi(self, nj, jposmass, jeps2):
        """
        Get individual gravitational potential due j-particles.
        """
        ni = self.n
        iposmass = self.stack_fields(('pos', 'mass'))
        ieps2 = self.eps2
        return gravitation.set_phi(ni, iposmass, ieps2,
                                   nj, jposmass, jeps2)

    def get_acc(self, nj, jposmass, jeps2):
        """
        Get individual gravitational acceleration due j-particles.
        """
        ni = self.n
        iposmass = self.stack_fields(('pos', 'mass'))
        ieps2 = self.eps2
        return gravitation.set_acc(ni, iposmass, ieps2,
                                   nj, jposmass, jeps2)

    def get_tstep(self, nj, jposmass, jveleps2, eta):
        """
        Get individual time-step due j-particles.
        """
        ni = self.n
        iposmass = self.stack_fields(('pos', 'mass'))
        iveleps2 = self.stack_fields(('vel', 'eps2'))
        return eta / gravitation.set_tstep(ni, iposmass, iveleps2,
                                           nj, jposmass, jveleps2, eta/2)

    def get_acctstep(self, jobj, eta):   # XXX: deprecated!
        """
        Get individual gravitational acceleration and time-step due j-particles.
        """
        if jobj.n:
            ret = gravitation.set_acctstep(self, jobj, eta/2)
            return (ret[0], eta / ret[1])
        return (0.0, 1.0)


    ### evolve

    def evolve_current_time(self, tau):
        """
        Evolves current time by tau.
        """
        self.t_curr += tau

    def evolve_position(self, tau):
        """
        Evolves position in time.
        """
        self.pos += tau * self.vel

    def evolve_velocity(self, tau):
        """
        Evolves velocity in time.
        """
        self.vel += tau * self.acc


########## end of file ##########
