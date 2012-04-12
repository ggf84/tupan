#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
from collections import namedtuple
import numpy as np
from .pbase import Pbase
from ..lib.interactor import interact


__all__ = ["BlackHole"]


class BlackHole(Pbase):
    """
    A base class for BlackHoles.
    """

    dtype = [# common attributes
             ("key", "u8"),
             ("mass", "f8"),
             ("pos", "3f8"),
             ("vel", "3f8"),
             ("acc", "3f8"),
             ("phi", "f8"),
             ("eps2", "f8"),
             ("tcurr", "f8"),
             ("tnext", "f8"),
             # specific attributes
             ("pnacc", "3f8"),
             ("spin", "3f8"),
             ("radius", "f8"),
             # auxiliary attributes
             ("tstep", "f8"),
	     ("pncorrection_energy", "f8"),
	     ("pncorrection_linear_momentum", "3f8"),
	     ("pncorrection_angular_momentum", "3f8"),
            ]

    def __init__(self, n=0):
        self.pncorrection_center_of_mass_position = np.zeros(3, "f8")
        self.pncorrection_center_of_mass_velocity = np.zeros(3, "f8")
        super(BlackHole, self).__init__(self.dtype, n)


    #
    # specific attributes
    #

    ### pnacc

    @property
    def pnacc(self):
        return self.data['pnacc']

    @pnacc.setter
    def pnacc(self, values):
        self.data['pnacc'] = values

    @pnacc.deleter
    def pnacc(self):
        raise NotImplementedError()

    ### spin

    @property
    def spin(self):
        return self.data['spin']

    @spin.setter
    def spin(self, values):
        self.data['spin'] = values

    @spin.deleter
    def spin(self):
        raise NotImplementedError()


    ### radius

    @property
    def radius(self):
        return self.data['radius']

    @radius.setter
    def radius(self, values):
        self.data['radius'] = values

    @radius.deleter
    def radius(self):
        raise NotImplementedError()


    #
    # auxiliary attributes
    #

    ### tstep

    @property
    def tstep(self):
        return self.data['tstep']

    @tstep.setter
    def tstep(self, values):
        self.data['tstep'] = values

    @tstep.deleter
    def tstep(self):
        raise NotImplementedError()


    ### pncorrection_energy

    @property
    def pncorrection_energy(self):
        return self.data['pncorrection_energy']

    @pncorrection_energy.setter
    def pncorrection_energy(self, values):
        self.data['pncorrection_energy'] = values

    @pncorrection_energy.deleter
    def pncorrection_energy(self):
        raise NotImplementedError()


    ### pncorrection_linear_momentum

    @property
    def pncorrection_linear_momentum(self):
        return self.data['pncorrection_linear_momentum']

    @pncorrection_linear_momentum.setter
    def pncorrection_linear_momentum(self, values):
        self.data['pncorrection_linear_momentum'] = values

    @pncorrection_linear_momentum.deleter
    def pncorrection_linear_momentum(self):
        raise NotImplementedError()


    ### pncorrection_angular_momentum

    @property
    def pncorrection_angular_momentum(self):
        return self.data['pncorrection_angular_momentum']

    @pncorrection_angular_momentum.setter
    def pncorrection_angular_momentum(self, values):
        self.data['pncorrection_angular_momentum'] = values

    @pncorrection_angular_momentum.deleter
    def pncorrection_angular_momentum(self):
        raise NotImplementedError()


    #
    # specific methods
    #

    ### evolve quantities due to post-newtonian corrections

    def evolve_velocity_due_to_pncorrection(self, tstep):
        """
        Evolves velocity in time due to post-newtonian correction.
        """
        self.vel += tstep * self.pnacc

    def evolve_center_of_mass_position_due_to_pncorrection(self, tstep):
        """
        Evolves center of mass position in time due to post-newtonian correction.
        """
        comp_jump = tstep * self.pncorrection_linear_momentum.sum(0)
        self.pncorrection_center_of_mass_position += comp_jump

    def evolve_center_of_mass_velocity_due_to_pncorrection(self, tstep):
        """
        Evolves center of mass velocity in time due to post-newtonian correction.
        """
        pnforce = -(self.mass * self.pnacc.T).T
        comv_jump = tstep * pnforce.sum(0)
        self.pncorrection_center_of_mass_velocity += comv_jump

    def evolve_energy_due_to_pncorrection(self, tstep):
        """
        Evolves energy in time due to post-newtonian correction.
        """
        pnforce = -(self.mass * self.pnacc.T).T
        e_jump = tstep * (self.vel * pnforce).sum(1)
        self.pncorrection_energy += e_jump

    def evolve_linear_momentum_due_to_pncorrection(self, tstep):
        """
        Evolves linear momentum in time due to post-newtonian correction.
        """
        pnforce = -(self.mass * self.pnacc.T).T
        lm_jump = tstep * pnforce
        self.pncorrection_linear_momentum += lm_jump

    def evolve_angular_momentum_due_to_pncorrection(self, tstep):
        """
        Evolves angular momentum in time due to post-newtonian correction.
        """
        pnforce = -(self.mass * self.pnacc.T).T
        am_jump = tstep * np.cross(self.pos, pnforce)
        self.pncorrection_angular_momentum += am_jump


    #
    # auxiliary methods
    #

    def get_pn_correction_for_center_of_mass_position(self):
        return self.pncorrection_center_of_mass_position

    def get_pn_correction_for_center_of_mass_velocity(self):
        return self.pncorrection_center_of_mass_velocity

    def get_pn_correction_for_total_energy(self):
        return self.pncorrection_energy.sum(0)

    def get_pn_correction_for_total_linear_momentum(self):
        return self.pncorrection_linear_momentum.sum(0)

    def get_pn_correction_for_total_angular_momentum(self):
        return self.pncorrection_angular_momentum.sum(0)




    #
    # overridden methods
    #

    ### gravity

    def update_phi(self, objs):
        """
        Update the individual gravitational potential due to other particles.
        """
        self.phi = interact.phi_blackhole(self, objs)

    def update_acc(self, objs):
        """
        Update the individual acceleration due to other particles.
        """
        self.acc, self.pnacc = interact.acc_blackhole(self, objs)

    def update_acctstep(self, objs, eta):
        """
        Update the individual acceleration and time-steps due to other particles.
        """
        (self.acc, self.pnacc, self.tstep) = interact.acctstep_blackhole(self, objs, eta)

    def update_tstep(self, objs, eta):
        """
        Update the individual time-steps due to other particles.
        """
        self.tstep = interact.tstep_blackhole(self, objs, eta)


    ### evolve

    def evolve_vel(self, tstep):
        """
        Evolves velocity in time.
        """
        self.vel += tstep * self.acc
        self.evolve_velocity_due_to_pncorrection(tstep)








###############################################################################
### XXX: old BlackHole




















dtype = {"names":   ["id", "mass", "radius", "tstep", "eps2", "phi", "pos", "vel", "acc", "pnacc", "spin", "_energy_jump", "_linmom_jump", "_angmom_jump"],
         "formats": ["u8", "f8",   "f8",     "f8",    "f8",   "f8",  "3f8", "3f8", "3f8", "3f8",   "3f8",  "f8",     "3f8",          "3f8"]}

#fields = OrderedDict([("index", "u8"), ("mass", "f8"), ("eps2", "f8"),
#                      ("phi", "f8"), ("pos", "3f8"),
#                      ("vel", "3f8"), ("acc", "3f8"), ("spin", "3f8")])
##dtype = fields.items()
#dtype = {"names": fields.keys(), "formats": fields.values()}


Energies = namedtuple("Energies", ["kin", "pot", "tot", "vir"])


class oldBlackHole(Pbase):
    """
    A base class for BlackHole-type particles.
    """

    def __init__(self, numobjs=0):
        Pbase.__init__(self, numobjs, dtype)

        self._totalmass = None
        self._self_total_epot = 0.0
        self._com_pos_jump = np.zeros(3, dtype="f8")
        self._com_vel_jump = np.zeros(3, dtype="f8")


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
        return self._com_pos_jump

    def get_center_of_mass_vel(self):
        """
        Get the center-of-mass velocity.
        """
        mtot = self.get_total_mass()
        return (self.mass * self.vel.T).sum(1) / mtot

    def get_com_vel_jump(self):
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
        return self._linmom_jump

    def get_angmom(self):
        """
        Get the individual angular momentum.
        """
        return (self.mass * np.cross(self.pos, self.vel).T).T

    def get_angmom_jump(self):
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

    def set_acctstep(self, objs, eta):
        """
        Set the individual acceleration and timesteps due to other particles.
        """
        (iacc, ipnacc, itstep) = interact.acctstep_blackhole(self, objs, eta)
        self.acc[:] = iacc
        self.pnacc[:] = ipnacc
        self.tstep[:] = itstep

    def set_tstep(self, objs, eta):
        """
        Set the individual timesteps due to other particles.
        """
        itstep = interact.tstep_blackhole(self, objs, eta)
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
        self._com_pos_jump += tstep * self._linmom_jump.sum(0) / self.get_total_mass()

    def evolve_com_vel_jump(self, tstep):
        """
        Evolves center of mass velocity jump due to an external force.
        """
        pnforce = -(self.mass * self.pnacc.T).T
        self._com_vel_jump += tstep * pnforce.sum(0) / self.get_total_mass()

    def evolve_energy_jump(self, tstep):
        """
        Evolves energy jump due to an external force.
        """
        pnforce = -(self.mass * self.pnacc.T).T
        self._energy_jump += tstep * (self.vel * pnforce).sum(1)

    def evolve_linmom_jump(self, tstep):
        """
        Evolves linear momentum jump due to an external force.
        """
        pnforce = -(self.mass * self.pnacc.T).T
        self._linmom_jump += tstep * pnforce

    def evolve_angmom_jump(self, tstep):
        """
        Evolves angular momentum jump due to an external force.
        """
        pnforce = -(self.mass * self.pnacc.T).T
        self._angmom_jump += tstep * np.cross(self.pos, pnforce)


########## end of file ##########
