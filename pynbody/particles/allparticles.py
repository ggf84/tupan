#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import print_function
import sys
import copy
import traceback
from collections import namedtuple
import numpy as np
from .sph import Sph
from .body import Body
from .blackhole import BlackHole
from ..lib.gravity import gravitation
from ..lib.utils.timing import timings


__all__ = ["Particles"]


ALL_PARTICLE_TYPES = ["sph", "body", "blackhole"]


Energies = namedtuple("Energies", ["kin", "pot", "tot", "vir"])


class Particles(dict):
    """
    This class holds the particle types in the simulation.
    """

    def __init__(self, types=None):
        """
        Initializer
        """
#        dict.__init__(self)

        self["sph"] = None #Sph()
        self["body"] = None #Body()
        self["blackhole"] = None #Blackhole()


        if types:
            if not isinstance(types, dict):
                print("types should be a dict instance", file=sys.stderr)

            if "sph" in types and types["sph"] > 0:
                self["sph"] = Sph(types["sph"])

            if "body" in types and types["body"] > 0:
                self["body"] = Body(types["body"])

            if "blackhole" in types and types["blackhole"] > 0:
                self["blackhole"] = BlackHole(types["blackhole"])

        self._totalmass = None

        super(Particles, self).__init__()


    def get_nbody(self):
        nb = 0
        for obj in self.values():
            if obj:
                nb += len(obj)
        return nb


    # Total Mass

    def get_total_mass(self):
        """
        Get the total mass of the whole system of particles.
        """
        total_mass = 0.0
        for obj in self.values():
            if obj:
                total_mass += obj.get_total_mass()
        return total_mass


    # Center-of-Mass methods

    def get_center_of_mass_position(self):
        """
        Get the center-of-mass position.
        """
        com_pos = 0.0
        total_mass = 0.0
        for obj in self.values():
            if obj:
                mass = obj.get_total_mass()
                total_mass += mass
                pos = obj.get_center_of_mass_position()
                com_pos += pos * mass
                if hasattr(obj, "get_pn_correction_for_center_of_mass_position"):
                    com_pos += obj.get_pn_correction_for_center_of_mass_position()
        return (com_pos / total_mass)

    def get_center_of_mass_velocity(self):
        """
        Get the center-of-mass velocity.
        """
        com_vel = 0.0
        total_mass = 0.0
        for obj in self.values():
            if obj:
                mass = obj.get_total_mass()
                total_mass += mass
                vel = obj.get_center_of_mass_velocity()
                com_vel += vel * mass
                if hasattr(obj, "get_pn_correction_for_center_of_mass_velocity"):
                    com_vel += obj.get_pn_correction_for_center_of_mass_velocity()
        return (com_vel / total_mass)

    def correct_center_of_mass(self):
        """
        Correct the center-of-mass to origin of coordinates.
        """
        com_pos = self.get_center_of_mass_position()
        com_vel = self.get_center_of_mass_velocity()
        for obj in self.values():
            if obj:
                obj.pos -= com_pos
                obj.vel -= com_vel


    # Momentum methods

    def get_total_linear_momentum(self):
        """
        Get the total linear momentum for the whole system of particles.
        """
        lin_mom = 0.0
        for obj in self.values():
            if obj:
                lin_mom += obj.get_total_linear_momentum()
                if hasattr(obj, "get_pn_correction_for_total_linear_momentum"):
                    lin_mom += obj.get_pn_correction_for_total_linear_momentum()
        return lin_mom

    def get_total_angular_momentum(self):
        """
        Get the total angular momentum for the whole system of particles.
        """
        ang_mom = 0.0
        for obj in self.values():
            if obj:
                ang_mom += obj.get_total_angular_momentum()
                if hasattr(obj, "get_pn_correction_for_total_angular_momentum"):
                    ang_mom += obj.get_pn_correction_for_total_angular_momentum()
        return ang_mom



    # Energy methods

    def get_total_kinetic_energy(self):
        """
        Get the total kinectic energy for the whole system of particles.
        """
        ke = 0.0
        for obj in self.values():
            if obj:
                ke += obj.get_total_kinetic_energy()
                if hasattr(obj, "get_pn_correction_for_total_energy"):
                    ke += obj.get_pn_correction_for_total_energy()
        return ke

    def get_total_potential_energy(self):
        """
        Get the total potential energy for the whole system of particles.
        """
        pe = 0.0
        for obj in self.values():
            if obj:
                pe += obj.get_total_potential_energy()     ### :FIXME: (when include BHs) ###
#                value = obj.get_total_potential_energy()
#                pe += 0.5*(value + obj._self_total_epot)
        return pe

    def get_total_etot(self):
        ekin = self.get_total_ekin()
        epot = self.get_total_epot()
        etot = ekin + epot
        return etot

    def get_total_evir(self):
        ekin = self.get_total_ekin()
        epot = self.get_total_epot()
        evir = ekin + (ekin + epot)
        return evir

    def get_energies(self):       ### is it necessary? ###
        """
        Get the energies ("kin", "pot", "tot", "vir") for each particle type.
        """
        energies = {}
        for (key, obj) in self.iteritems():
            if obj:
                energies[key] = obj.get_total_energies()
            else:
                energies[key] = None
        return energies

    def get_energy_jump(self):     ### :FIXME: ###      ### is it necessary? ###
        energy_jump = {}
        for (key, obj) in self.iteritems():
            if hasattr(obj, "get_total_energy_jump"):
                energy_jump[key] = obj.get_total_energy_jump()
            else:
                energy_jump[key] = None
        return energy_jump

    def get_total_energies(self):       ### is it necessary? ###
        """
        Get the total energies ("kin", "pot", "tot", "vir") for the whole
        system of particles.
        """
        ekin = 0.0
        epot = 0.0
        for (key, energy) in self.get_energies().iteritems():
            if energy is not None:
                ekin += energy.kin
                epot += 0.5*(energy.pot + self[key]._self_total_epot)
#                XXX: To fix it: sum_epot = BoBo + BHBH + (BoBH + BHBo) + ...
#                   : Now it looks OK, but it's better verify it again
#                   : when other particles types are implemented.
        etot = ekin + epot
        evir = ekin + etot
        energies = Energies(ekin, epot, etot, evir)
        return energies

    def get_total_energy_jump(self):     ### :FIXME: ###
        energy_jump = self.get_energy_jump()
        return sum((value for value in energy_jump.itervalues()
                    if value is not None))


    # Gravity methods

    @timings
    def update_phi(self, objs):
        """
        Update the individual gravitational potential due to other particles.
        """
        for iobj in self.values():
            if iobj:
                iphi = 0.0
                for jobj in objs.values():
                    if jobj:
                        ret = gravitation.newtonian.set_phi(iobj, jobj)
                        iphi += ret
                iobj.phi = iphi

    @timings
    def update_acc(self, objs):
        """
        Update the individual gravitational acceleration due to other particles.
        """
        for iobj in self.values():
            if iobj:
                iacc = 0.0
                for jobj in objs.values():
                    if jobj:
                        ret = gravitation.newtonian.set_acc(iobj, jobj)
                        iacc += ret
                iobj.acc = iacc

    @timings
    def update_pnacc(self, objs):
        """
        Update the individual post-newtonian gravitational acceleration due to other particles.
        """
        for iobj in self.values():
            if iobj:
                ipnacc = 0.0
                if hasattr(iobj, "pnacc"):
                    for jobj in objs.values():
                        if jobj:
                            if hasattr(jobj, "pnacc"):
                                ret = gravitation.post_newtonian.set_acc(iobj, jobj)
                                ipnacc += ret
                    iobj.pnacc = ipnacc

    @timings
    def update_acctstep(self, objs, eta):
        """
        Update the individual gravitational acceleration and time-steps due to other particles.
        """
        eta_2 = eta/2
        for iobj in self.values():
            if iobj:
                iacc = 0.0
                iomega = 0.0
                for jobj in objs.values():
                    if jobj:
                        ret = gravitation.newtonian.set_acctstep(iobj, jobj, eta_2)
                        iacc += ret[0]
                        iomega = np.maximum(iomega, ret[1])
                iobj.acc = iacc
                iobj.tstep = eta/iomega


    @timings
    def update_tstep(self, objs, eta):
        """
        Update the individual time-steps due to other particles.
        """
        eta_2 = eta/2
        for iobj in self.values():
            if iobj:
                iomega = 0.0
                for jobj in objs.values():
                    if jobj:
                        ret = gravitation.newtonian.set_tstep(iobj, jobj, eta_2)
                        iomega = np.maximum(iomega, ret)
                iobj.tstep = eta/iomega




    # Miscelaneous methods

    def any(self):
        has_obj = False
        for obj in self.itervalues():
            if obj:
                has_obj = True
        return has_obj

    def copy(self):
        return copy.deepcopy(self)

    def append(self, data):
        for (key, obj) in data.iteritems():
            if obj:
                if self[key]:
#                    tmp = self[key][:]
                    tmp = self[key].copy()
                    tmp.append(obj)
                    self[key] = tmp
                else:
                    self[key] = obj


    def set_members(self, data):
        """
        Set particle member types.
        """
        try:
            if isinstance(data, Body):
                self["body"] = data
            elif isinstance(data, BlackHole):
                self["blackhole"] = data
            elif isinstance(data, Sph):
                self["sph"] = data
            else:
                raise TypeError("Unsupported particle type.")
        except TypeError as msg:
            traceback.print_list(
                    traceback.extract_stack(
                            sys.exc_info()[2].tb_frame.f_back, None), None)
            print("TypeError: {0}".format(msg))
            sys.exit(-1)


########## end of file ##########
