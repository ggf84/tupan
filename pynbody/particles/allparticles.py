#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import print_function
import sys
import traceback
import numpy as np
from collections import namedtuple
from .sph import Sph
from .body import Body
from .blackhole import BlackHole
from pynbody.lib import gravity



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
        gravity.build_kernels()

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

        self._phi = {"sph": {"body": 0.0, "blackhole": 0.0},
                     "body": {"sph": 0.0, "blackhole": 0.0},
                     "blackhole": {"sph": 0.0, "body": 0.0}}



    # Total Mass

    def update_total_mass(self):
        mass = 0.0
        for obj in self.itervalues():
            if obj:
                obj.update_total_mass()
                mass += obj._totalmass
        self._totalmass = mass

    def get_total_mass(self):
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

    def get_center_of_mass_vel(self):
        """
        Get the center-of-mass velocity.
        """
        comVel = 0.0
        for obj in self.itervalues():
            if obj:
                mass = obj.get_total_mass()
                vel = obj.get_center_of_mass_pos()
                comVel += vel * mass
        return (comVel / self.get_total_mass())

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
        return linmom

    def get_angmom(self):
        """
        Get the total angular momentum for each particle type.
        """
        angmom = {}
        for (key, obj) in self.iteritems():
            if obj:
                angmom[key] = obj.get_total_angmom()
        return angmom

    def get_total_linmom(self):
        """
        Get the total linear momentum for the whole system of particles.
        """
        linmom = 0.0
        for value in self.get_linmom().itervalues():
            linmom += value
        return linmom

    def get_total_angmom(self):
        """
        Get the total angular momentum for the whole system of particles.
        """
        angmom = 0.0
        for value in self.get_angmom().itervalues():
            angmom += value
        return angmom


    # Energy methods

    def get_energies(self):
        """
        Get the energies ("kin", "pot", "tot", "vir") for each particle type.
        """
        energies = {}
        for (key, obj) in self.iteritems():
            if obj:
                energies[key] = obj.get_total_energies()
        return energies

    def get_total_energies(self):
        """
        Get the total energies ("kin", "pot", "tot", "vir") for the whole
        system of particles.
        """
        ekin = 0.0
        epot = 0.0
        for value in self.get_energies().itervalues():
            ekin += value.kin
            epot += value.pot   # XXX: To fix it: BoBo + BHBH + (BoBH + BHBo) +...
        etot = ekin + epot
        evir = ekin + etot
        energies = Energies(ekin, epot, etot, evir)
        return energies


    # Gravity methods

    def set_phi(self, objs):
        for (key, obj) in self.iteritems():
            if obj:
                obj.set_phi(objs)

    def _accumulate_phi_for(self, iobj):
        phi = 0
        for (key, jobj) in self.iteritems():
            if jobj:
                if isinstance(iobj, Sph):
                    pass                # XXX: to be implemented.
                if isinstance(iobj, Body):
                    phi += gravity.newtonian.set_phi(iobj, jobj)
                if isinstance(iobj, BlackHole):
                    pass                # XXX: to be implemented.

        return phi


#    def for_each_set_phi(self, objs):
#        for key in ALL_PARTICLE_TYPES:
#            self._phi_sph[key] = self.set_phi_sph_from(objs[key])
#            self._phi_body[key] = self.set_phi_body_from(objs[key])
#            self._phi_blackhole[key] = self.set_phi_blackhole_from(objs[key])



    def set_acc(self, objs):
        rhostep = {}
        for (key, obj) in self.iteritems():
            if obj:
                rhostep[key] = obj.set_acc(objs)
            else:
                rhostep[key] = None
        return rhostep

    def _accumulate_acc_for(self, iobj):
        acc = 0
        rhostep = 0
        for (key, jobj) in self.iteritems():
            if jobj:
                if isinstance(iobj, Sph):
                    pass                # XXX: to be implemented.
                if isinstance(iobj, Body):
                    ret = gravity.newtonian.set_acc(iobj, jobj)
                    ret[:,3] = np.sqrt(ret[:,3]/(len(jobj)-1))
                    acc += ret[:,:3]
                    rhostep += ret[:,3]
                if isinstance(iobj, BlackHole):
                    pass                # XXX: to be implemented.

        return (acc, rhostep)




    # Miscelaneous methods

    def any(self):
        has_obj = False
        for obj in self.itervalues():
            if obj:
                has_obj = True
        return has_obj

    def copy(self):
        ret = self.__class__()
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


########## end of file ##########
