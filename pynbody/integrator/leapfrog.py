#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
from ..lib.utils.timing import timings


__all__ = ["LeapFrog"]



_coefs = [1.0]

#_coefs = [1.3512071919596575,
#         -1.7024143839193150,
#          1.3512071919596575]

#_coefs = [0.4144907717943757,
#          0.4144907717943757,
#         -0.6579630871775028,
#          0.4144907717943757,
#          0.4144907717943757]

#_coefs = [0.3221375960817983,
#          0.5413165481700432,
#         -0.7269082885036829,
#          0.5413165481700432,
#          0.3221375960817983]

#_coefs = [0.7845136104775573,
#          0.23557321335935813,
#         -1.177679984178871,
#          1.3151863206839112,
#         -1.177679984178871,
#          0.23557321335935813,
#          0.7845136104775573]

#_coefs = [0.1867,
#          0.5554970237124784,
#          0.12946694891347535,
#         -0.8432656233877346,
#          0.9432033015235617,
#         -0.8432656233877346,
#          0.12946694891347535,
#          0.5554970237124784,
#          0.1867]


class LeapFrog(object):
    """

    """
    def __init__(self, eta, current_time, particles, coefs=_coefs):
        self.eta = eta
        self.current_time = current_time
        self.coefs = coefs
        self.particles = particles

        omega = self.particles.set_acc(particles, 0.0)

        self.tau = self.get_tau(omega)
        self.tstep = self.tau


    @timings
    def gather(self):
        return self.particles


    def get_tau(self, omega):
        omega_sum = 0.0
        for (key, value) in omega.items():
            if value is not None:
                omega_sum = max(omega_sum, value.max())
        tau = float(self.eta / omega_sum**0.5)
        return tau


    @timings
    def drift(self, step):
        """

        """
        self.tstep += step
        self.current_time += step
        for (key, obj) in self.particles.iteritems():
            if hasattr(obj, "evolve_pos"):
                obj.evolve_pos(step)
            if hasattr(obj, "evolve_com_pos_jump"):
                obj.evolve_com_pos_jump(step)


    @timings
    def forceDKD(self, jparticles, step):
        """

        """
        prev_acc = {}
        prev_pnacc = {}
        for (key, obj) in self.particles.items():
            if hasattr(obj, "acc"):
                prev_acc[key] = obj.acc.copy()
            if hasattr(obj, "pnacc"):
                prev_pnacc[key] = obj.pnacc.copy()

        omega = self.particles.set_acc(jparticles, step)

        for (key, obj) in self.particles.items():
            if hasattr(obj, "acc"):
                obj.acc[:] = 2 * obj.acc - prev_acc[key]
            if hasattr(obj, "pnacc"):
                obj.pnacc[:] = 2 * obj.pnacc - prev_pnacc[key]

        tau = self.get_tau(omega)
        return tau


    @timings
    def kick(self, step):
        """

        """
        for (key, obj) in self.particles.iteritems():
            if hasattr(obj, "pnacc"):
                external_force = -(obj.mass * obj.pnacc.T).T
                if hasattr(obj, "evolve_com_vel_jump"):
                    obj.evolve_com_vel_jump(0.5 * step, external_force)
                if hasattr(obj, "evolve_linmom_jump"):
                    obj.evolve_linmom_jump(0.5 * step, external_force)
                if hasattr(obj, "evolve_angmom_jump"):
                    obj.evolve_angmom_jump(0.5 * step, external_force)
                if hasattr(obj, "evolve_energy_jump"):
                    obj.evolve_energy_jump(0.5 * step, external_force)
            if hasattr(obj, "evolve_vel"):
                obj.evolve_vel(0.5 * step)

        nextstep = self.forceDKD(self.gather(), step)

        for (key, obj) in self.particles.iteritems():
            if hasattr(obj, "evolve_vel"):
                obj.evolve_vel(0.5 * step)
            if hasattr(obj, "pnacc"):
                external_force = -(obj.mass * obj.pnacc.T).T
                if hasattr(obj, "evolve_com_vel_jump"):
                    obj.evolve_com_vel_jump(0.5 * step, external_force)
                if hasattr(obj, "evolve_linmom_jump"):
                    obj.evolve_linmom_jump(0.5 * step, external_force)
                if hasattr(obj, "evolve_angmom_jump"):
                    obj.evolve_angmom_jump(0.5 * step, external_force)
                if hasattr(obj, "evolve_energy_jump"):
                    obj.evolve_energy_jump(0.5 * step, external_force)

        return nextstep


    def stepDKD(self, methcoef):
        """

        """
        currstep = self.tau
        self.drift(0.5 * methcoef * currstep)
        nextstep = self.kick(methcoef * currstep)
        self.drift(0.5 * methcoef * currstep)
        self.tau = nextstep


    @timings
    def step(self):
        """

        """
        self.tstep = 0.0
        ncoefs = len(self.coefs)
        for coef in self.coefs:
            self.stepDKD(ncoefs*coef)



    # Pickle-related methods

    def __getstate__(self):
        sdict = self.__dict__.copy()
        return sdict

    def __setstate__(self, sdict):
        self.__dict__.update(sdict)
        self.particles = self.particles.copy()


########## end of file ##########
