#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import copy
import numpy as np


__all__ = ['Pbase']


class Pbase(object):
    """

    """
    def __init__(self, dtype, n):
        self.data = np.zeros(n, dtype)

    #
    # common basic methods
    #

    def __repr__(self):
        return '{0}({1})'.format(self.__class__.__name__, self.data)

    def __len__(self):
        return len(self.data)

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

    ### key

    @property
    def key(self):
        return self.data['key']

    @key.setter
    def key(self, values):
        self.data['key'] = values

    @key.deleter
    def key(self):
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


    ### tcurr

    @property
    def tcurr(self):
        return self.data['tcurr']

    @tcurr.setter
    def tcurr(self, values):
        self.data['tcurr'] = values

    @tcurr.deleter
    def tcurr(self):
        raise NotImplementedError()


    ### tnext

    @property
    def tnext(self):
        return self.data['tnext']

    @tnext.setter
    def tnext(self, values):
        self.data['tnext'] = values

    @tnext.deleter
    def tnext(self):
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
        raise NotImplementedError()

    def update_acc(self, jobj):
        raise NotImplementedError()

    def update_acctstep(self, jobj, eta):
        raise NotImplementedError()

    def update_tstep(self, jobj, eta):
        raise NotImplementedError()


    ### evolve

    def evolve_position(self, tstep):
        """
        Evolves position in time.
        """
        self.pos += tstep * self.vel

    def evolve_velocity(self, tstep):
        """
        Evolves velocity in time.
        """
        self.vel += tstep * self.acc


########## end of file ##########
