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
    dtype = None
    zero = None

    def __init__(self, n):
        self.data = self.zero if n == 0 else np.zeros(n, self.dtype)


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

    def __len__(self):
        return len(self.data)
    n = property(__len__)

    def __getitem__(self, index):
        data = self.data[index]
        cp = copy.copy(self)
        cp.data = data
        return cp

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
        jpos = jobj.pos
        jmass = jobj.mass
        jeps2 = jobj.eps2
        self.phi = self.get_phi(nj, jpos, jmass, jeps2)

    def update_acc(self, jobj):
        nj = jobj.n
        jpos = jobj.pos
        jmass = jobj.mass
        jeps2 = jobj.eps2
        self.acc = self.get_acc(nj, jpos, jmass, jeps2)

    def update_timestep(self, jobj, eta):
        nj = jobj.n
        jpos = jobj.pos
        jmass = jobj.mass
        jvel = jobj.vel
        jeps2 = jobj.eps2
        self.dt_next = self.get_tstep(nj, jpos, jmass, jvel, jeps2, eta)

    def update_acc_and_timestep(self, jobj, eta):   # XXX: deprecated!
        ret = iobj.get_acctstep(jobj, eta)
        self.acc = ret[0]
        self.dt_next = ret[1]


    def get_phi(self, nj, jpos, jmass, jeps2):
        """
        Get individual gravitational potential due j-particles.
        """
        ni = self.n
        ipos = self.pos
        imass = self.mass
        ieps2 = self.eps2
        return gravitation.set_phi(ni, ipos, imass, ieps2,
                                   nj, jpos, jmass, jeps2)

    def get_acc(self, nj, jpos, jmass, jeps2):
        """
        Get individual gravitational acceleration due j-particles.
        """
        ni = self.n
        ipos = self.pos
        imass = self.mass
        ieps2 = self.eps2
        return gravitation.set_acc(ni, ipos, imass, ieps2,
                                   nj, jpos, jmass, jeps2)

    def get_tstep(self, nj, jpos, jmass, jvel, jeps2, eta):
        """
        Get individual time-step due j-particles.
        """
        ni = self.n
        ipos = self.pos
        imass = self.mass
        ivel = self.vel
        ieps2 = self.eps2
        return eta / gravitation.set_tstep(ni, ipos, imass, ivel, ieps2,
                                           nj, jpos, jmass, jvel, jeps2, eta/2)

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


    ### state

    def get_state(self):
        attrs = list(self.data.dtype.names[:10])
        return self.data[attrs]

    def set_state(self, state):
        self.data = np.zeros(len(state), dtype=self.dtype)
        for name in state.dtype.names:
            if name in self.data.dtype.names:
                self.data[name] = state[name]


########## end of file ##########
