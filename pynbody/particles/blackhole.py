#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
from collections import (namedtuple, OrderedDict)
from .pbase import Pbase


__all__ = ["BlackHole"]


fields = OrderedDict([("index", "u8"), ("mass", "f8"), ("eps2", "f8"),
                      ("phi", "f8"), ("stepdens", "2f8"), ("pos", "3f8"),
                      ("vel", "3f8"), ("acc", "3f8"), ("spin", "3f8")])
#dtype = fields.items()
dtype = {"names": fields.keys(), "formats": fields.values()}


Energies = namedtuple("Energies", ["kin", "pot", "tot", "vir"])


class BlackHole(Pbase):
    """
    A base class for BlackHole-type particles.
    """

    def __init__(self, numobjs=0):
        Pbase.__init__(self, numobjs, dtype)

        self._totalmass = None
        self._own_total_epot = 0.0

        self._pnacc = None
        self._energy_jump = 0.0
        self._com_pos_jump = 0.0
        self._com_vel_jump = 0.0
        self._linmom_jump = 0.0
        self._angmom_jump = 0.0


    # Total Mass

    def update_total_mass(self):
        """
        Updates the total mass to the current sum.
        """
        self._totalmass = np.sum(self.mass)

    def get_total_mass(self):
        """
        Get the total mass.
        """
        if self._totalmass is None:
            self.update_total_mass()
        return self._totalmass


    # Center-of-Mass methods

    def get_center_of_mass_pos(self):
        """
        Get the center-of-mass position.
        """
        mtot = self.get_total_mass()
        return (self.mass * self.pos.T).sum(1) / mtot + self._com_pos_jump

    def get_center_of_mass_vel(self):
        """
        Get the center-of-mass velocity.
        """
        mtot = self.get_total_mass()
        return (self.mass * self.vel.T).sum(1) / mtot + self._com_vel_jump

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
        return (self.mass * self.vel.T).T + self._linmom_jump

    def get_angmom(self):
        """
        Get the individual angular momentum.
        """
        return (self.mass * np.cross(self.pos, self.vel).T).T + self._angmom_jump

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
        Get the individual kinetic energy.
        """
        return 0.5 * self.mass * (self.vel**2).sum(1) + self._energy_jump

    def get_epot(self):
        """
        Get the individual potential energy.
        """
        return self.mass * self.phi

    def get_etot(self):
        """
        Get the individual "kinetic + potential" energy.
        """
        return self.get_ekin() + self.get_epot()

    def get_energies(self):
        """
        Get the individual energies ("kin", "pot", "tot", "vir").
        """
        ekin = self.get_ekin()
        epot = self.get_epot()
        etot = ekin + epot
        evir = ekin + etot
        energies = Energies(ekin, epot, etot, evir)
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
        return np.sum(self.get_epot()) - self._own_total_epot

    def get_total_etot(self):
        """
        Get the total "kinetic + potential" energy.
        """
        return self.get_total_ekin() + self.get_total_epot()

    def get_total_energies(self):
        """
        Get the total energies ("kin", "pot", "tot", "vir").
        """
        ekin = self.get_total_ekin()
        epot = self.get_total_epot()
        etot = ekin + epot
        evir = ekin + etot
        energies = Energies(ekin, epot, etot, evir)
        return energies


    # Gravity methods

    def set_phi(self, objs):
        """
        Set the individual gravitational potential due to other particles.
        """
        (self.phi[:], self._own_total_epot) = objs._accumulate_phi_for(self)

    def set_acc(self, objs):
        """
        Set the individual acceleration due to other particles.
        """
        if self._pnacc == None:
            self._pnacc = np.zeros_like(self.acc)
        (self.acc[:], self._pnacc[:], rhostep) = objs._accumulate_acc_for(self)
        return rhostep


    # Evolving methods

    def evolve_pos(self, dpos):
        """
        Evolves position by dpos.
        """
        self.pos += dpos

    def evolve_vel(self, dvel):
        """
        Evolves velocity by dvel.
        """
        self.vel += dvel

    def evolve_energy_jump(self, denergy_jump):
        """
        Evolves energy jump by denergy_jump.
        """
        self._energy_jump += denergy_jump

    def evolve_com_pos_jump(self, dcom_pos_jump):
        """
        Evolves center of mass position jump by dcom_pos_jump.
        """
        self._com_pos_jump += dcom_pos_jump

    def evolve_com_vel_jump(self, dcom_vel_jump):
        """
        Evolves center of mass velocity jump by dcom_vel_jump.
        """
        self._com_vel_jump += dcom_vel_jump

    def evolve_linmom_jump(self, dlinmom_jump):
        """
        Evolves linear momentum jump by dlinmom_jump.
        """
        self._linmom_jump += dlinmom_jump

    def evolve_angmom_jump(self, dangmom_jump):
        """
        Evolves angular momentum jump by dangmom_jump.
        """
        self._angmom_jump += dangmom_jump


########## end of file ##########
