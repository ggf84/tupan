#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
from collections import namedtuple
from pynbody.particles.pbase import Pbase
import pynbody.lib.gravity as gravity


__all__ = ['Body']


fields = dict([('index', 'u8'), ('mass', 'f8'), ('eps2', 'f8'),
               ('phi', 'f8'), ('stepdens', '2f8'), ('pos', '3f8'),
               ('vel', '3f8'), ('acc', '3f8')])
#dtype = fields.items()
dtype = {'names': fields.keys(), 'formats': fields.values()}


class Body(Pbase):
    """
    A base class for Body-type particles.
    """

    def __init__(self, numobjs=0):
        Pbase.__init__(self, numobjs, dtype)


    # Total Mass

    def get_total_mass(self):
        return np.sum(self.mass)


    # Center-of-Mass methods

    def get_center_of_mass_pos(self):
        """
        Get the center-of-mass position.
        """
        return (self.mass * self.pos.T).sum(1) / self.get_total_mass()

    def get_center_of_mass_vel(self):
        """
        Get the center-of-mass velocity.
        """
        return (self.mass * self.vel.T).sum(1) / self.get_total_mass()

    def reset_center_of_mass(self):
        """
        Reset the center-of-mass to origin.
        """
        self.pos -= self.get_center_of_mass_pos()
        self.vel -= self.get_center_of_mass_vel()


    # Momentum methods

    def get_linmom(self):
        """
        Get the individual linear momentum.
        """
        return (self.mass * self.vel.T).T

    def get_angmom(self):
        """
        Get the individual angular momentum.
        """
        return (self.mass * np.cross(self.pos, self.vel).T).T

    def get_total_linmom(self):
        """
        Get the total linear momentum.
        """
        return self.get_linmom().sum(0)

    def get_total_angmom(self):
        """
        Get the total angular momentum.
        """
        return self.get_angmom().sum(0)


    # Energy methods

    def get_ekin(self):
        """
        Get the individual kinetic energy in the center-of-mass frame.
        """
        vcm = self.get_center_of_mass_vel()
        return 0.5 * self.mass * ((self.vel - vcm)**2).sum(1)

    def get_epot(self):
        """
        Get the individual potential energy.
        """
        return self.mass * self.phi

    def get_etot(self):
        """
        Get the individual kinetic+potential energy.
        """
        return self.get_ekin() + self.get_epot()

    def get_energies(self):
        """
        Get the individual energies 'kin', 'pot', 'tot', 'vir'.
        """
        ekin = self.get_ekin()
        epot = self.get_epot()
        etot = ekin + epot
        Energies = namedtuple('Energies', ['kin', 'pot', 'tot', 'vir'])
        energies = Energies(ekin, epot, etot, 2*ekin+epot)
        return energies

    def get_total_ekin(self):
        """
        Get the total kinetic energy.
        """
        return np.sum(self.get_ekin())

    def get_total_epot(self):
        """
        Get the total potential energy.
        """
        return 0.5 * np.sum(self.get_epot())

    def get_total_etot(self):
        """
        Get the total kinetic+potential energy.
        """
        return self.get_total_ekin() + self.get_total_epot()

    def get_total_energies(self):
        """
        Get the total energies 'kin', 'pot', 'tot', 'vir'.
        """
        ekin = self.get_total_ekin()
        epot = self.get_total_epot()
        etot = ekin + epot
        TotEnergies = namedtuple('TotEnergies', ['kin', 'pot', 'tot', 'vir'])
        totenergies = TotEnergies(ekin, epot, etot, 2*ekin+epot)
        return totenergies


    # Gravity methods

    def set_phi(self, objs):
        """
        Set the individual gravitational potential due to other bodies.
        """
        self.phi[:] = gravity.newtonian.set_phi(self, objs)

    def set_acc(self, objs):
        """
        Set the individual acceleration due to other bodies.
        """
        _acc = gravity.newtonian.set_acc(self, objs)
        self.acc[:] = _acc[:,:3]
        self.stepdens[:,0] = np.sqrt(_acc[:,3]/(len(objs)-1))


    # Evolve methods

    def drift(self, tau):
        self.pos += tau * self.vel

    def kick(self, tau):
        self.vel += tau * self.acc


########## end of file ##########
