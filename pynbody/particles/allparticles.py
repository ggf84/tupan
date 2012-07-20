#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import print_function
import sys
import copy
import numpy as np
from .sph import Sph
from .body import Body
from .blackhole import BlackHole
from ..lib import gravity
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Particles"]


ALL_PARTICLE_TYPES = ["sph", "body", "blackhole"]

@decallmethods(timings)
class Particles(dict):
    """
    This class holds the particle types in the simulation.
    """
    def __init__(self):
        """
        Initializer
        """
        super(Particles, self).__init__(
                                        sph=Sph(),
                                        body=Body(),
                                        blackhole=BlackHole(),
                                       )
        self.__dict__.update(self)


    #
    # miscellaneous methods
    #

    def __hash__(self):
        i = None
        for obj in self.values():
            if i is None:
                i = hash(obj)
            else:
                i ^= hash(obj)
        return i


    def __repr__(self):
        fmt = self.__class__.__name__+'{'
        for key, obj in self.items():
            fmt += '\n{0},'.format(obj)
        fmt += '\n}'
        return fmt


    def __len__(self):
        return sum([obj.n for obj in self.values()])
    n = property(__len__)


    @property
    def empty(self):
        return self.__class__()


    def copy(self):
        return copy.deepcopy(self)


    def append(self, objs):
        if objs.n:
            if isinstance(objs, Particles):
                for (key, obj) in objs.items():
                    if obj.n:
                        self[key].append(obj)
            else:
                name = objs.__class__.__name__.lower()
                self[name].append(objs)


    def stack_fields(self, attrs, pad=-1):
        arrays = [obj.stack_fields(attrs, pad) for obj in self.values() if obj.n]
        return np.concatenate(arrays)


    def select(self, function, attr):
        subset = self.empty
        for obj in self.values():
            if obj.n:
                selection = function(getattr(obj, attr))
                subset.append(obj[selection])
        return subset


    #
    # common attributes
    #

    ### ...


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
        gravity.phi.set_args(self, objs)
        gravity.phi.run()
        result = gravity.phi.get_result()
        for iobj in self.values():
            if iobj.n:
                iobj.phi = result[:iobj.n]
                result = result[iobj.n:]

    def update_acc(self, objs):
        """
        Update the individual gravitational acceleration due to other particles.
        """
        gravity.acc.set_args(self, objs)
        gravity.acc.run()
        result = gravity.acc.get_result()
        for iobj in self.values():
            if iobj.n:
                iobj.acc = result[:iobj.n]
                result = result[iobj.n:]

    def update_tstep(self, objs, eta):
        """
        Update the individual time-steps due to other particles.
        """
        gravity.tstep.set_args(self, objs, eta/2)
        gravity.tstep.run()
        result = gravity.tstep.get_result()
        for iobj in self.values():
            if iobj.n:
                iobj.dt_next = result[:iobj.n]
                result = result[iobj.n:]

    def update_pnacc(self, objs, pn_order, clight):
        """
        Update the individual post-newtonian gravitational acceleration due to other particles.
        """
        ni = self.blackhole.n
        nj = objs.blackhole.n
        if ni and nj:
            gravity.pnacc.set_args(self.blackhole, objs.blackhole, pn_order, clight)
            gravity.pnacc.run()
            result = gravity.pnacc.get_result()
            for iobj in self.values():
                if iobj.n:
                    if hasattr(iobj, "pnacc"):
                        iobj.pnacc = result[:iobj.n]
                        result = result[iobj.n:]

    def update_acc_jerk(self, objs):
        """
        Update the individual gravitational acceleration and jerk due to other particles.
        """
        gravity.acc_jerk.set_args(self, objs)
        gravity.acc_jerk.run()
        (acc, jerk) = gravity.acc_jerk.get_result()
        for iobj in self.values():
            if iobj.n:
                iobj.acc = acc[:iobj.n]
                iobj.jerk = jerk[:iobj.n]
                acc = acc[iobj.n:]
                jerk = jerk[iobj.n:]


    ### nstep

    def update_nstep(self):
        for obj in self.values():
            if obj.n:
                obj.update_nstep()

    def update_t_curr(self, tau):
        for obj in self.values():
            if obj.n:
                obj.update_t_curr(tau)


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
                min_value = min(min_value, np.abs(obj.dt_prev).min())
        return min_value

    def min_dt_next(self):
        """

        """
        min_value = sys.float_info.max
        for obj in self.values():
            if obj.n:
                min_value = min(min_value, np.abs(obj.dt_next).min())
        return min_value


    def max_dt_prev(self):
        """

        """
        max_value = 0.0
        for obj in self.values():
            if obj.n:
                max_value = max(max_value, np.abs(obj.dt_prev).max())
        return max_value

    def max_dt_next(self):
        """

        """
        max_value = 0.0
        for obj in self.values():
            if obj.n:
                max_value = max(max_value, np.abs(obj.dt_next).max())
        return max_value


    def power_averaged_dt_prev(self, power):
        """

        """
        n = 0
        average = 0.0
        for obj in self.values():
            if obj.n:
                average += (obj.dt_prev**power).sum()
                n += obj.n
        average = (average / n)**(1.0/power)
        return average


    def power_averaged_dt_next(self, power):
        """

        """
        n = 0
        average = 0.0
        for obj in self.values():
            if obj.n:
                average += (obj.dt_next**power).sum()
                n += obj.n
        average = (average / n)**(1.0/power)
        return average


########## end of file ##########
