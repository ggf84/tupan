#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
from collections import namedtuple
#from collections import (namedtuple, OrderedDict)
import numpy as np
from .pbase import Pbase
from ..lib.interactor import interact


__all__ = ["BlackHole"]


dtype = {"names":   ["id", "mass", "radius", "tstep", "eps2", "phi", "pos", "vel", "acc", "pnacc", "spin"],
         "formats": ["u8", "f8",   "f8",     "f8",    "f8",   "f8",  "3f8", "3f8", "3f8", "3f8",   "3f8"]}

#fields = OrderedDict([("index", "u8"), ("mass", "f8"), ("eps2", "f8"),
#                      ("phi", "f8"), ("pos", "3f8"),
#                      ("vel", "3f8"), ("acc", "3f8"), ("spin", "3f8")])
##dtype = fields.items()
#dtype = {"names": fields.keys(), "formats": fields.values()}


Energies = namedtuple("Energies", ["kin", "pot", "tot", "vir"])


class BlackHole(Pbase):
    """
    A base class for BlackHole-type particles.
    """

    def __init__(self, numobjs=0):
        Pbase.__init__(self, numobjs, dtype)

        self._totalmass = None
        self._self_total_epot = 0.0

        self._energy_jump = None
        self._com_pos_jump = None
        self._com_vel_jump = None
        self._linmom_jump = None
        self._angmom_jump = None


    # Total Mass

    def update_total_mass(self):
        """
        Updates the total mass to the current sum.
        """
        self._totalmass = float(np.sum(self.mass))

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
        return (self.mass * self.pos.T).sum(1) / mtot

    def get_com_pos_jump(self):
        if self._com_pos_jump == None:
            self._com_pos_jump = np.zeros(3, dtype="f8")
        return self._com_pos_jump

    def get_center_of_mass_vel(self):
        """
        Get the center-of-mass velocity.
        """
        mtot = self.get_total_mass()
        return (self.mass * self.vel.T).sum(1) / mtot

    def get_com_vel_jump(self):
        if self._com_vel_jump == None:
            self._com_vel_jump = np.zeros(3, dtype="f8")
        return self._com_vel_jump

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

    def get_linmom_jump(self):
        if self._linmom_jump == None:
            self._linmom_jump = np.zeros_like(self.acc)
        return self._linmom_jump

    def get_angmom(self):
        """
        Get the individual angular momentum.
        """
        return (self.mass * np.cross(self.pos, self.vel).T).T

    def get_angmom_jump(self):
        if self._angmom_jump == None:
            self._angmom_jump = np.zeros_like(self.acc)
        return self._angmom_jump

    def get_total_linmom(self):
        """
        Get the total linear momentum.
        """
        return self.get_linmom().sum(0)

    def get_total_linmom_jump(self):
        return self.get_linmom_jump().sum(0)

    def get_total_angmom(self):
        """
        Get the total angular momentum.
        """
        return self.get_angmom().sum(0)

    def get_total_angmom_jump(self):
        return self.get_angmom_jump().sum(0)


    # Energy methods

    def get_ekin(self):
        """
        Get the individual kinetic energy.
        """
        return 0.5 * self.mass * (self.vel**2).sum(1)

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

    def get_energy_jump(self):
        if self._energy_jump == None:
            self._energy_jump = np.zeros_like(self.phi)
        return self._energy_jump

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
        return float(np.sum(self.get_ekin()))

    def get_total_epot(self):
        """
        Get the total potential energy.
        """
        return float(np.sum(self.get_epot())) - self._self_total_epot

    def get_total_etot(self):
        """
        Get the total "kinetic + potential" energy.
        """
        return self.get_total_ekin() + self.get_total_epot()

    def get_total_energy_jump(self):
        return float(np.sum(self.get_energy_jump()))

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
        (iphi, self_phi) = interact.phi_blackhole(self, objs)
        self.phi[:] = iphi
        self._self_total_epot = 0.5 * float(np.sum(self.mass * self_phi))

    def set_acc(self, objs):
        """
        Set the individual acceleration due to other particles.
        """
        (iacc, ipnacc) = interact.acc_blackhole(self, objs)
        self.acc[:] = iacc
        self.pnacc[:] = ipnacc

    def set_acctstep(self, objs, tau):
        """
        Set the individual acceleration and timesteps due to other particles.
        """
        (iacc, ipnacc, iomega) = interact.acctstep_blackhole(self, objs, tau)
        self.acc[:] = iacc
        self.pnacc[:] = ipnacc
        return iomega

    def set_tstep(self, objs, eta, old_tstep):
        """
        Set the individual timesteps due to other particles.
        """
        itstep = interact.tstep_blackhole(self, objs, eta, old_tstep)
        self.tstep[:] = itstep


    # Evolving methods

    def evolve_pos(self, tstep):
        """
        Evolves position in time.
        """
        self.pos += tstep * self.vel

    def evolve_vel(self, tstep):
        """
        Evolves velocity in time.
        """
        self.vel += tstep * (self.acc + self.pnacc)

    def evolve_com_pos_jump(self, tstep):
        """
        Evolves center of mass position jump due to an external force.
        """
        if self._com_pos_jump == None:
            self._com_pos_jump = np.zeros(3, dtype="f8")
        self._com_pos_jump += tstep * self._com_vel_jump

    def evolve_com_vel_jump(self, tstep, external_force):
        """
        Evolves center of mass velocity jump due to an external force.
        """
        if self._com_vel_jump == None:
            self._com_vel_jump = np.zeros(3, dtype="f8")
        self._com_vel_jump += tstep * external_force.sum(0) / self.get_total_mass()

    def evolve_energy_jump(self, tstep, external_force):
        """
        Evolves energy jump due to an external force.
        """
        if self._energy_jump == None:
            self._energy_jump = np.zeros_like(self.phi)
        self._energy_jump += tstep * (self.vel * external_force).sum(1)

    def evolve_linmom_jump(self, tstep, external_force):
        """
        Evolves linear momentum jump due to an external force.
        """
        if self._linmom_jump == None:
            self._linmom_jump = np.zeros_like(self.pnacc)
        self._linmom_jump += tstep * external_force

    def evolve_angmom_jump(self, tstep, external_force):
        """
        Evolves angular momentum jump due to an external force.
        """
        if self._angmom_jump == None:
            self._angmom_jump = np.zeros_like(self.pnacc)
        self._angmom_jump += tstep * np.cross(self.pos, external_force)


########## end of file ##########
