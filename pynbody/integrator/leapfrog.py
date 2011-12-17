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
        self.particles = particles

        self.gamma = 1.0
        methcoef = len(self.coefs) * self.coefs[0]
        rhostep = self.particles.set_acc(particles)

        vdotf = 0.0
        for (key, obj) in self.particles.iteritems():
            if hasattr(obj, "acc"):
                vdotf += 2 * np.sum(obj.mass * (obj.vel*obj.acc).sum(1))
        omega = self.get_omega(rhostep)
        tau = self.get_tau(omega)
        dtau_dt = self.get_dtau_dt(tau, omega, vdotf) #* methcoef
        symm_factor = (dtau_dt + np.sqrt(dtau_dt**2 + 4.0))/2.0
#        symm_factor = 1.0/(1.0 - 0.5 * dtau_dt)
        self.tau = tau * float(symm_factor)


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
    def drift(self, step):
        """

        """
        self.tstep += step
        self.time += step
        for (key, obj) in self.particles.iteritems():
            if hasattr(obj, "evolve_pos"):
                obj.evolve_pos(step * obj.vel)
            if hasattr(obj, "evolve_com_pos_jump"):
                obj.evolve_com_pos_jump(step * obj._com_vel_jump)


    @timings
    def kick(self, step):
        """

        """
        for (key, obj) in self.particles.iteritems():
            if hasattr(obj, "evolve_vel"):
                obj.evolve_vel(step * self.dvel[key])


    @timings
    def forceDKD(self, jparticles, methcoef, currstep):
        """

        """
        vdotf = 0.0
        fullstep = methcoef * currstep
        rhostep = self.particles.set_acc(jparticles)

        for (key, obj) in self.particles.iteritems():
            if hasattr(obj, "acc"):
                g0 = self.dvel[key].copy()
                self.dvel[key][:] = 2 * currstep * obj.acc - self.dvel[key]
                g1 = self.dvel[key].copy()

                v12 = obj.vel + 0.25 * methcoef * (g1-g0)

                if hasattr(obj, "_pnacc"):
                    force_ext = -(obj.mass * obj._pnacc.T).T
                    if hasattr(obj, "evolve_energy_jump"):
                        energy_jump = (v12 * force_ext).sum(1)
                        obj.evolve_energy_jump(fullstep * energy_jump)
                    if hasattr(obj, "evolve_com_vel_jump"):
                        mtot = obj.get_total_mass()
                        com_vel_jump = force_ext.sum(0) / mtot
                        obj.evolve_com_vel_jump(fullstep * com_vel_jump)
                    if hasattr(obj, "evolve_linmom_jump"):
                        linmom_jump = force_ext
                        obj.evolve_linmom_jump(fullstep * linmom_jump)
                    if hasattr(obj, "evolve_angmom_jump"):
                        angmom_jump = np.cross(obj.pos, force_ext)
                        obj.evolve_angmom_jump(fullstep * angmom_jump)

                vdotf += 2 * np.sum(obj.mass * (v12*obj.acc).sum(1))

        omega = self.get_omega(rhostep)
        tau = self.get_tau(omega)
        dtau_dt = self.get_dtau_dt(tau, omega, vdotf) #* methcoef
        symm_factor = (dtau_dt + np.sqrt(dtau_dt**2 + 4.0))/2.0
#        symm_factor = 1.0/(1.0 - 0.5 * dtau_dt)

        nexttau = tau * float(symm_factor)
        return nexttau


    def stepDKD(self, methcoef):
        """

        """
        currstep = self.tau
        self.drift(0.5 * methcoef * currstep)
        self.kick(0.5 * methcoef)
        nextstep = self.forceDKD(self.gather(), methcoef, currstep)
        self.kick(0.5 * methcoef)
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
