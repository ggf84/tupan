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

    @property
    def n(self):
        return sum([obj.n for obj in self.values()])


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

    def stack_fields(self, attrs, pad=-1):
        arrays = [obj.stack_fields(attrs, pad) for obj in self.values() if obj.n]
        return recf.stack_arrays(arrays).reshape((self.n,-1)).squeeze()


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
#        nj = objs.n
#        jdata = objs.stack_fields(('pos', 'mass', 'vel', 'eps2'))
#        for iobj in self.values():
#            if iobj.n:
#                iobj.phi = iobj.get_phi(nj, jdata)


        ni = self.n
        idata = self.stack_fields(('pos', 'mass', 'vel', 'eps2'))
        nj = objs.n
        jdata = objs.stack_fields(('pos', 'mass', 'vel', 'eps2'))

        gravity.phi.set_arg('IN', 0, ni)
        gravity.phi.set_arg('IN', 1, idata)
        gravity.phi.set_arg('IN', 2, nj)
        gravity.phi.set_arg('IN', 3, jdata)
        gravity.phi.set_arg('OUT', 4, (ni,))
        gravity.phi.global_size = ni
        gravity.phi.run()
        result = gravity.phi.get_result()[0]

        for iobj in self.values():
            if iobj.n:
                iobj.phi = result[:iobj.n]
                result = result[iobj.n:]


    def update_acc(self, objs):
        """
        Update the individual gravitational acceleration due to other particles.
        """
#        nj = objs.n
#        jdata = objs.stack_fields(('pos', 'mass', 'vel', 'eps2'))
#        for iobj in self.values():
#            if iobj.n:
#                iobj.acc = iobj.get_acc(nj, jdata)


        ni = self.n
        idata = self.stack_fields(('pos', 'mass', 'vel', 'eps2'))
        nj = objs.n
        jdata = objs.stack_fields(('pos', 'mass', 'vel', 'eps2'))

        gravity.acc.set_arg('IN', 0, ni)
        gravity.acc.set_arg('IN', 1, idata)
        gravity.acc.set_arg('IN', 2, nj)
        gravity.acc.set_arg('IN', 3, jdata)
        gravity.acc.set_arg('OUT', 4, (ni, 4))
        gravity.acc.global_size = ni
        gravity.acc.run()
        result = gravity.acc.get_result()[0][:,:3]

        for iobj in self.values():
            if iobj.n:
                iobj.acc = result[:iobj.n]
                result = result[iobj.n:]


    def update_acc_jerk(self, objs):
        """
        Update the individual gravitational acceleration and jerk due to other particles.
        """
#        nj = objs.n
#        jdata = objs.stack_fields(('pos', 'mass', 'vel', 'eps2'))
#        for iobj in self.values():
#            if iobj.n:
#                (iobj.acc, iobj.jerk) = iobj.get_acc_jerk(nj, jdata)


        ni = self.n
        idata = self.stack_fields(('pos', 'mass', 'vel', 'eps2'))
        nj = objs.n
        jdata = objs.stack_fields(('pos', 'mass', 'vel', 'eps2'))

        gravity.acc_jerk.set_arg('IN', 0, ni)
        gravity.acc_jerk.set_arg('IN', 1, idata)
        gravity.acc_jerk.set_arg('IN', 2, nj)
        gravity.acc_jerk.set_arg('IN', 3, jdata)
        gravity.acc_jerk.set_arg('OUT', 4, (ni, 8))
        gravity.acc_jerk.global_size = ni
        gravity.acc_jerk.run()
        result = gravity.acc_jerk.get_result()[0]

        for iobj in self.values():
            if iobj.n:
                iobj.acc = result[:iobj.n][:,:3]
                iobj.jerk = result[:iobj.n][:,4:7]
                result = result[iobj.n:]


    def update_timestep(self, objs, eta):
        """
        Update the individual time-steps due to other particles.
        """
#        nj = objs.n
#        jdata = objs.stack_fields(('pos', 'mass', 'vel', 'eps2'))
#        for iobj in self.values():
#            if iobj.n:
#                iobj.dt_next = iobj.get_tstep(nj, jdata, eta)


        ni = self.n
        idata = self.stack_fields(('pos', 'mass', 'vel', 'eps2'))
        nj = objs.n
        jdata = objs.stack_fields(('pos', 'mass', 'vel', 'eps2'))

        gravity.tstep.set_arg('IN', 0, ni)
        gravity.tstep.set_arg('IN', 1, idata)
        gravity.tstep.set_arg('IN', 2, nj)
        gravity.tstep.set_arg('IN', 3, jdata)
        gravity.tstep.set_arg('IN', 4, eta/2)
        gravity.tstep.set_arg('OUT', 5, (ni,))
        gravity.tstep.global_size = ni
        gravity.tstep.run()
        result = gravity.tstep.get_result()[0]

        for iobj in self.values():
            if iobj.n:
                iobj.dt_next = eta / result[:iobj.n]
                result = result[iobj.n:]


    def update_pnacc(self, objs, pn_order, clight):
        """
        Update the individual post-newtonian gravitational acceleration due to other particles.
        """
#        objs = objs['blackhole']
#        nj = objs.n
#        jdata = objs.stack_fields(('pos', 'mass', 'vel'), pad=8)
#        for iobj in self.values():
#            if iobj.n:
#                if hasattr(iobj, "pnacc"):
#                    if nj:
#                        iobj.pnacc = iobj.get_pnacc(nj, jdata, pn_order, clight)
#                    else:
#                        iobj.pnacc = 0.0


        ni = self['blackhole'].n
        idata = self['blackhole'].stack_fields(('pos', 'mass', 'vel'), pad=8)
        nj = objs['blackhole'].n
        jdata = objs['blackhole'].stack_fields(('pos', 'mass', 'vel'), pad=8)

        gravity.pnacc.set_arg('IN', 0, ni)
        gravity.pnacc.set_arg('IN', 1, idata)
        gravity.pnacc.set_arg('IN', 2, nj)
        gravity.pnacc.set_arg('IN', 3, jdata)

        clight = gravity.Clight(pn_order, clight)
        gravity.pnacc.set_arg('IN', 4, clight.pn_order)
        gravity.pnacc.set_arg('IN', 5, clight.inv1)
        gravity.pnacc.set_arg('IN', 6, clight.inv2)
        gravity.pnacc.set_arg('IN', 7, clight.inv3)
        gravity.pnacc.set_arg('IN', 8, clight.inv4)
        gravity.pnacc.set_arg('IN', 9, clight.inv5)
        gravity.pnacc.set_arg('IN', 10, clight.inv6)
        gravity.pnacc.set_arg('IN', 11, clight.inv7)

        gravity.pnacc.set_arg('OUT', 12, (ni, 4))
        gravity.pnacc.global_size = ni
        gravity.pnacc.run()
        result = gravity.pnacc.get_result()[0][:,:3]

        for iobj in self.values():
            if iobj.n:
                if hasattr(iobj, "pnacc"):
                    iobj.pnacc = result[:iobj.n]
                    result = result[iobj.n:]


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
