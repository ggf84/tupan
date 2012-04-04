#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import print_function
import sys
import traceback
from collections import namedtuple
import numpy as np
from .sph import Sph
from .body import Body
from .blackhole import BlackHole
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
        dict.__init__(self)

        self["sph"] = None
        self["body"] = None
        self["blackhole"] = None


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


    def get_nbody(self):
        nb = 0
        for obj in self.values():
            if obj:
                nb += len(obj)
        return nb


    # Total Mass

    def update_total_mass(self):
        """
        Updates the total mass of the whole system of particles
        to the current sum.
        """
        mass = 0.0
        for obj in self.itervalues():
            if obj:
                obj.update_total_mass()
                mass += obj._totalmass
        self._totalmass = mass

    def get_total_mass(self):
        """
        Get the total mass of the whole system of particles.
        """
        if self._totalmass is None:
            self.update_total_mass()
        return self._totalmass


    # Center-of-Mass methods

    def get_center_of_mass_pos(self):
        """
        Get the center-of-mass position.
        """
        comPos = 0.0
        for obj in self.itervalues():
            if obj:
                mass = obj.get_total_mass()
                pos = obj.get_center_of_mass_pos()
                comPos += pos * mass
        return (comPos / self.get_total_mass())

    def get_com_pos_jump(self):
        comPosJump = sum((obj.get_com_pos_jump() * obj.get_total_mass()
                          for obj in self.itervalues()
                          if hasattr(obj, "get_com_pos_jump")))
        return (comPosJump / self.get_total_mass())

    def get_center_of_mass_vel(self):
        """
        Get the center-of-mass velocity.
        """
        comVel = 0.0
        for obj in self.itervalues():
            if obj:
                mass = obj.get_total_mass()
                vel = obj.get_center_of_mass_vel()
                comVel += vel * mass
        return (comVel / self.get_total_mass())

    def get_com_vel_jump(self):
        comVelJump = sum((obj.get_com_vel_jump() * obj.get_total_mass()
                          for obj in self.itervalues()
                          if hasattr(obj, "get_com_vel_jump")))
        return (comVelJump / self.get_total_mass())

    def reset_center_of_mass(self):
        """
        Reset the center-of-mass to origin.
        """
        comPos = self.get_center_of_mass_pos()
        comVel = self.get_center_of_mass_vel()
        for obj in self.itervalues():
            if obj:
                obj.pos -= comPos
                obj.vel -= comVel


    # Momentum methods

    def get_linmom(self):
        """
        Get the total linear momentum for each particle type.
        """
        linmom = {}
        for (key, obj) in self.iteritems():
            if obj:
                linmom[key] = obj.get_total_linmom()
            else:
                linmom[key] = None
        return linmom

    def get_linmom_jump(self):
        linmom = {}
        for (key, obj) in self.iteritems():
            if hasattr(obj, "get_total_linmom_jump"):
                linmom[key] = obj.get_total_linmom_jump()
            else:
                linmom[key] = None
        return linmom

    def get_angmom(self):
        """
        Get the total angular momentum for each particle type.
        """
        angmom = {}
        for (key, obj) in self.iteritems():
            if obj:
                angmom[key] = obj.get_total_angmom()
            else:
                angmom[key] = None
        return angmom

    def get_angmom_jump(self):
        angmom = {}
        for (key, obj) in self.iteritems():
            if hasattr(obj, "get_total_angmom_jump"):
                angmom[key] = obj.get_total_angmom_jump()
            else:
                angmom[key] = None
        return angmom

    def get_total_linmom(self):
        """
        Get the total linear momentum for the whole system of particles.
        """
        linmom = 0.0
        for value in self.get_linmom().itervalues():
            if value is not None:
                linmom += value
        return linmom

    def get_total_linmom_jump(self):
        linmom_jump = self.get_linmom_jump()
        return sum((value for value in linmom_jump.itervalues()
                    if value is not None))

    def get_total_angmom(self):
        """
        Get the total angular momentum for the whole system of particles.
        """
        angmom = 0.0
        for value in self.get_angmom().itervalues():
            if value is not None:
                angmom += value
        return angmom

    def get_total_angmom_jump(self):
        angmom_jump = self.get_angmom_jump()
        return sum((value for value in angmom_jump.itervalues()
                    if value is not None))


    # Energy methods

    def get_total_ekin(self):
        """
        Get the total kinectic energy for the whole system of particles.
        """
        ekin = 0.0
        for (key, obj) in self.items():
            if obj:
                ekin += obj.get_total_ekin()
        return ekin

    def get_total_epot(self):
        """
        Get the total potential energy for the whole system of particles.
        """
        epot = 0.0
        for (key, obj) in self.items():
            if obj:
                value = obj.get_total_epot()
                epot += 0.5*(value + obj._self_total_epot)
        return epot

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

    def get_energies(self):
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

    def get_energy_jump(self):
        energy_jump = {}
        for (key, obj) in self.iteritems():
            if hasattr(obj, "get_total_energy_jump"):
                energy_jump[key] = obj.get_total_energy_jump()
            else:
                energy_jump[key] = None
        return energy_jump

    def get_total_energies(self):
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

    def get_total_energy_jump(self):
        energy_jump = self.get_energy_jump()
        return sum((value for value in energy_jump.itervalues()
                    if value is not None))


    # Gravity methods

    @timings
    def set_phi(self, objs):
        for (key, obj) in self.iteritems():
            if obj:
                obj.set_phi(objs)

    @timings
    def set_acc(self, objs):
        for (key, obj) in self.iteritems():
            if obj:
                obj.set_acc(objs)

    @timings
    def set_acctstep(self, objs, tau):
        rhostep = {}
        for (key, obj) in self.iteritems():
            if obj:
                rhostep[key] = obj.set_acctstep(objs, tau)
            else:
                rhostep[key] = None
        return rhostep

    @timings
    def set_tstep(self, objs, eta, old_tstep):
        for (key, obj) in self.iteritems():
            if obj:
                obj.set_tstep(objs, eta, old_tstep)




    # Miscelaneous methods

    def any(self):
        has_obj = False
        for obj in self.itervalues():
            if obj:
                has_obj = True
        return has_obj

    def copy(self):
        ret = self.__class__()
        ret.__dict__.update(self.__dict__)
        for (key, obj) in self.iteritems():
            if obj:
                ret[key] = obj.copy()
        return ret

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



    # Pickle-related methods

    def __getstate__(self):
        sdict = self.__dict__.copy()
        return sdict

    def __setstate__(self, sdict):
        self.__dict__.update(sdict)
        self = self.copy()


########## end of file ##########
