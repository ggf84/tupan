#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import print_function
import sys
import copy
import numpy as np
from numpy.lib import recfunctions as recf
from .sph import Sph
from .body import Body
from .blackhole import BlackHole
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Particles"]


ALL_PARTICLE_TYPES = ["sph", "body", "blackhole"]

@decallmethods(timings)
class Particles(dict):
    """
    This class holds the particle types in the simulation.
    """
    def __init__(self, types=None):
        """
        Initializer
        """
        self["sph"] = Sph()
        self["body"] = Body()
        self["blackhole"] = BlackHole()

        if types:
            if not isinstance(types, dict):
                print("types should be a dict instance", file=sys.stderr)

            if "sph" in types and types["sph"] > 0:
                self["sph"] = Sph(types["sph"])

            if "body" in types and types["body"] > 0:
                self["body"] = Body(types["body"])

            if "blackhole" in types and types["blackhole"] > 0:
                self["blackhole"] = BlackHole(types["blackhole"])

        self.n = 0
        super(Particles, self).__init__()


    #
    # miscellaneous methods
    #

    def __repr__(self):
        fmt = self.__class__.__name__+'{'
        for key, obj in self.items():
            fmt += '\n{0},'.format(obj)
        fmt += '\n}'
        return fmt

    def copy(self):
        return copy.deepcopy(self)

    def append(self, objs):
        if isinstance(objs, Particles):
            for (key, obj) in objs.items():
                if obj.n:
                    self[key].append(obj)
        elif isinstance(objs, Body):
            self["body"].append(objs)
        elif isinstance(objs, BlackHole):
            self["blackhole"].append(objs)
        elif isinstance(objs, Sph):
            self["sph"].append(objs)

    def update_n(self):
        self.n = sum([obj.n for obj in self.values()])


    #
    # common attributes
    #

    @property
    def pos(self):
        pos = [obj.pos for obj in self.values() if obj.n]
        return recf.stack_arrays(pos).view(np.ndarray).reshape((-1,3))

    @property
    def vel(self):
        vel = [obj.vel for obj in self.values() if obj.n]
        return recf.stack_arrays(vel).view(np.ndarray).reshape((-1,3))

    @property
    def mass(self):
        mass = [obj.mass for obj in self.values() if obj.n]
        return recf.stack_arrays(mass).view(np.ndarray)

    @property
    def eps2(self):
        eps2 = [obj.eps2 for obj in self.values() if obj.n]
        return recf.stack_arrays(eps2).view(np.ndarray)


    #
    # common methods
    #

    ### total mass and center-of-mass

    def get_total_mass(self):
        """
        Get the total mass of the whole system of particles.
        """
        return sum([obj.get_total_mass() for obj in self.values() if obj.n])

    def get_center_of_mass_position(self):
        """
        Get the center-of-mass position.
        """
        com_pos = 0.0
        total_mass = 0.0
        for obj in self.values():
            if obj.n:
                mass = obj.get_total_mass()
                pos = obj.get_center_of_mass_position()
                com_pos += pos * mass
                total_mass += mass
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
            if obj.n:
                mass = obj.get_total_mass()
                vel = obj.get_center_of_mass_velocity()
                com_vel += vel * mass
                total_mass += mass
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
            if obj.n:
                obj.pos -= com_pos
                obj.vel -= com_vel


    ### linear momentum

    def get_total_linear_momentum(self):
        """
        Get the total linear momentum for the whole system of particles.
        """
        lin_mom = 0.0
        for obj in self.values():
            if obj.n:
                lin_mom += obj.get_total_linear_momentum()
                if hasattr(obj, "get_pn_correction_for_total_linear_momentum"):
                    lin_mom += obj.get_pn_correction_for_total_linear_momentum()
        return lin_mom


    ### angular momentum

    def get_total_angular_momentum(self):
        """
        Get the total angular momentum for the whole system of particles.
        """
        ang_mom = 0.0
        for obj in self.values():
            if obj.n:
                ang_mom += obj.get_total_angular_momentum()
                if hasattr(obj, "get_pn_correction_for_total_angular_momentum"):
                    ang_mom += obj.get_pn_correction_for_total_angular_momentum()
        return ang_mom


    ### kinetic energy

    def get_total_kinetic_energy(self):
        """
        Get the total kinectic energy for the whole system of particles.
        """
        ke = 0.0
        for obj in self.values():
            if obj.n:
                ke += obj.get_total_kinetic_energy()
                if hasattr(obj, "get_pn_correction_for_total_energy"):
                    ke += obj.get_pn_correction_for_total_energy()
        return ke


    ### potential energy

    def get_total_potential_energy(self):
        """
        Get the total potential energy for the whole system of particles.
        """
        pe = 0.0
        for obj in self.values():
            if obj.n:
                pe += obj.get_total_potential_energy()     ### :FIXME: (when include BHs) ###
#                value = obj.get_total_potential_energy()
#                pe += 0.5*(value + obj._self_total_epot)
        return pe


    ### gravity

    def update_phi(self, objs):
        """
        Update the individual gravitational potential due to other particles.
        """
        nj = objs.n
        jpos = objs.pos
        jmass = objs.mass
        jeps2 = objs.eps2
        for iobj in self.values():
            if iobj.n:
                iobj.phi = iobj.get_phi(nj, jpos, jmass, jeps2)

    def update_acc(self, objs):
        """
        Update the individual gravitational acceleration due to other particles.
        """
        nj = objs.n
        jpos = objs.pos
        jmass = objs.mass
        jeps2 = objs.eps2
        for iobj in self.values():
            if iobj.n:
                iobj.acc = iobj.get_acc(nj, jpos, jmass, jeps2)

    def update_timestep(self, objs, eta):
        """
        Update the individual time-steps due to other particles.
        """
        nj = objs.n
        jpos = objs.pos
        jmass = objs.mass
        jvel = objs.vel
        jeps2 = objs.eps2
        for iobj in self.values():
            if iobj.n:
                iobj.dt_next = iobj.get_tstep(nj, jpos, jmass, jvel, jeps2, eta)

    def update_pnacc(self, objs, pn_order, clight):
        """
        Update the individual post-newtonian gravitational acceleration due to other particles.
        """
        for iobj in self.values():
            if iobj.n:
                if hasattr(iobj, "pnacc"):
                    ipnacc = 0.0
                    for jobj in objs.values():
                        if hasattr(jobj, "pnacc"):
                            ipnacc += iobj.get_pnacc(jobj, pn_order, clight)
                    iobj.pnacc = ipnacc

    def update_acc_and_timestep(self, objs, eta):   # XXX: deprecated!
        """
        Update the individual gravitational acceleration and time-steps due to other particles.
        """
        for iobj in self.values():
            if iobj.n:
                iacc = 0.0
                itstep = 1.0
                for jobj in objs.values():
                    ret = iobj.get_acctstep(jobj, eta)
                    iacc += ret[0]
                    itstep = np.minimum(itstep, ret[1])
                iobj.acc = iacc
                iobj.dt_next = itstep


    ### prev/next timestep

    def set_dt_prev(self, tau):
        """

        """
        for obj in self.values():
            if obj.n:
                obj.dt_prev = tau

    def set_dt_next(self, tau):
        """

        """
        for obj in self.values():
            if obj.n:
                obj.dt_next = tau


    def min_dt_prev(self):
        """

        """
        min_value = sys.float_info.max
        for obj in self.values():
            if obj.n:
                min_value = min(min_value, obj.dt_prev.min())
        return min_value

    def min_dt_next(self):
        """

        """
        min_value = sys.float_info.max
        for obj in self.values():
            if obj.n:
                min_value = min(min_value, obj.dt_next.min())
        return min_value


    def max_dt_prev(self):
        """

        """
        max_value = 0.0
        for obj in self.values():
            if obj.n:
                max_value = max(max_value, obj.dt_prev.max())
        return max_value

    def max_dt_next(self):
        """

        """
        max_value = 0.0
        for obj in self.values():
            if obj.n:
                max_value = max(max_value, obj.dt_next.max())
        return max_value


    def mean_dt_prev(self):
        """

        """
        n = 0
        mean_value = 0.0
        for obj in self.values():
            if obj.n:
                mean_value += obj.dt_prev.sum()
                n += obj.n
        mean_value = mean_value / n
        return mean_value

    def mean_dt_next(self):
        """

        """
        n = 0
        mean_value = 0.0
        for obj in self.values():
            if obj.n:
                mean_value += obj.dt_next.sum()
                n += obj.n
        mean_value = mean_value / n
        return mean_value


    def harmonic_mean_dt_prev(self):
        """

        """
        n = 0
        harmonic_mean_value = 0.0
        for obj in self.values():
            if obj.n:
                harmonic_mean_value += (1 / obj.dt_prev).sum()
                n += obj.n
        harmonic_mean_value = n / harmonic_mean_value
        return harmonic_mean_value

    def harmonic_mean_dt_next(self):
        """

        """
        n = 0
        harmonic_mean_value = 0.0
        for obj in self.values():
            if obj.n:
                harmonic_mean_value += (1 / obj.dt_next).sum()
                n += obj.n
        harmonic_mean_value = n / harmonic_mean_value
        return harmonic_mean_value


########## end of file ##########
