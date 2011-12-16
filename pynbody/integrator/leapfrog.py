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
    def __init__(self, eta, time, particles, coefs=_coefs):
        self.eta = eta
        self.time = time
        self.tstep = eta
        self.coefs = coefs
        rhostep = particles.set_acc(particles)
        self.particles = particles

        self.gamma = 1.0

        omega = self.get_omega(rhostep)
        self.tau = self.get_tau(omega)

        self.dvel = {}
        for (key, obj) in self.particles.iteritems():
            if hasattr(obj, "acc"):
                self.dvel[key] = self.tau * obj.acc


    @timings
    def gather(self):
        return self.particles


    def get_omega(self, rhostep):
        omega = 0.0
        for (key, value) in rhostep.items():
            if value is not None:
                omega += value.sum()
        return omega

    def get_tau(self, omega):
        return float(self.eta / omega**self.gamma)

    def get_dtau_dt(self, tau, omega, vdotf):
        return -self.gamma * (tau/omega) * vdotf


    @timings
    def drift(self, stepcoef):
        """

        """
        varstep = 0.5 * stepcoef * self.tau
        self.time += varstep
        self.tstep += varstep
        for (key, obj) in self.particles.iteritems():
            if hasattr(obj, "evolve_pos"):
                obj.evolve_pos(varstep * obj.vel)
            if hasattr(obj, "evolve_com_pos_jump"):
                obj.evolve_com_pos_jump(varstep * obj._com_vel_jump)


    @timings
    def kick(self, stepcoef):
        """

        """
        for (key, obj) in self.particles.iteritems():
            if hasattr(obj, "evolve_vel"):
                obj.evolve_vel(0.5 * stepcoef * self.dvel[key])


    @timings
    def forceDKD(self, jparticles, stepcoef):
        """

        """
        rhostep = self.particles.set_acc(jparticles)
        omega = self.get_omega(rhostep)

        self.tau = self.get_tau(omega)
        varstep = stepcoef * self.tau

        vdotf = 0.0

        for (key, obj) in self.particles.iteritems():
            if hasattr(obj, "acc"):
                g0 = self.dvel[key].copy()
                self.dvel[key][:] = 2 * self.tau * obj.acc - self.dvel[key]
                g1 = self.dvel[key].copy()

                v12 = obj.vel + 0.25 * stepcoef * (g1-g0)

                if hasattr(obj, "_pnacc"):
                    force_ext = -(obj.mass * obj._pnacc.T).T
                    if hasattr(obj, "evolve_energy_jump"):
                        energy_jump = (v12 * force_ext).sum(1)
                        obj.evolve_energy_jump(varstep * energy_jump)
                    if hasattr(obj, "evolve_com_vel_jump"):
                        mtot = obj.get_total_mass()
                        com_vel_jump = force_ext.sum(0) / mtot
                        obj.evolve_com_vel_jump(varstep * com_vel_jump)
                    if hasattr(obj, "evolve_linmom_jump"):
                        linmom_jump = force_ext
                        obj.evolve_linmom_jump(varstep * linmom_jump)
                    if hasattr(obj, "evolve_angmom_jump"):
                        angmom_jump = np.cross(obj.pos, force_ext)
                        obj.evolve_angmom_jump(varstep * angmom_jump)

                vdotf += 2 * np.sum(obj.mass * (v12*obj.acc).sum(1))

        dtau_dt = self.get_dtau_dt(varstep, omega, vdotf)

#        symm_factor = 1.0/(1.0 - 0.5 * dtau_dt)
        symm_factor = (dtau_dt + np.sqrt(dtau_dt**2 + 4.0))/2.0

        self.tau *= float(symm_factor)


    def stepDKD(self, stepcoef):
        """

        """
        self.drift(stepcoef)
        self.kick(stepcoef)
        self.forceDKD(self.gather(), stepcoef)
        self.kick(stepcoef)
        self.drift(stepcoef)


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
