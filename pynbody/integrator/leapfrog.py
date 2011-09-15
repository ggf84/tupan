#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np

from ggf84decor import selftimer


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
        self.coefs = coefs
        particles.set_phi(particles)
        rhostep = particles.set_acc(particles)
        self.particles = particles
        self.e0 = self.particles.get_total_energies()

#        varstep = 1.0
        varstep = 0.5 / (-self.e0.pot)
        self.dvel = {}
        for (key, obj) in self.particles.iteritems():
            if hasattr(obj, "acc"):
                self.dvel[key] = varstep * obj.acc


    @selftimer
    def gather(self):
        return self.particles


    @selftimer
    def drift(self, stepcoef):
        """

        """
        e = self.particles.get_total_energies()
        ejump = self.particles.get_total_energy_jump()
#        varstep = 1.0
        varstep = 0.5 / ((e.kin - self.e0.tot) + ejump)
        tau = 0.5 * stepcoef * self.eta * varstep
        self.time += tau
        for (key, obj) in self.particles.iteritems():
            if hasattr(obj, "evolve_pos"):
                obj.evolve_pos(tau * obj.vel)
            if hasattr(obj, "evolve_com_pos_jump"):
                obj.evolve_com_pos_jump(tau * obj._com_vel_jump)


    @selftimer
    def kick(self, stepcoef):
        """

        """
        tau = 0.5 * stepcoef * self.eta
        for (key, obj) in self.particles.iteritems():
            if hasattr(obj, "evolve_vel"):
                obj.evolve_vel(tau * self.dvel[key])


    @selftimer
    def forceDKD(self, jparticles, stepcoef):
        """

        """
        rhostep = self.particles.set_acc(jparticles)
        self.particles.set_phi(jparticles)

        e = self.particles.get_total_energies()
#        varstep = 1.0
        varstep = 0.5 / (-e.pot)
        tau = stepcoef * self.eta * varstep
        for (key, obj) in self.particles.iteritems():
            if hasattr(obj, "acc"):
                g0 = self.dvel[key].copy()
                self.dvel[key][:] = 2 * varstep * obj.acc - self.dvel[key]
                g1 = self.dvel[key].copy()
                if hasattr(obj, "_pnacc"):
                    force_ext = -(obj.mass * obj._pnacc.T).T
                    if hasattr(obj, "evolve_energy_jump"):
                        v12 = obj.vel + 0.25 * tau * (g1-g0) / varstep
                        energy_jump = (v12 * force_ext).sum(1)
                        obj.evolve_energy_jump(tau * energy_jump)
                    if hasattr(obj, "evolve_com_vel_jump"):
                        mtot = obj.get_total_mass()
                        com_vel_jump = force_ext.sum(0) / mtot
                        obj.evolve_com_vel_jump(tau * com_vel_jump)
                    if hasattr(obj, "evolve_linmom_jump"):
                        linmom_jump = force_ext
                        obj.evolve_linmom_jump(tau * linmom_jump)
                    if hasattr(obj, "evolve_angmom_jump"):
                        angmom_jump = np.cross(obj.pos, force_ext)
                        obj.evolve_angmom_jump(tau * angmom_jump)


    def stepDKD(self, stepcoef):
        """

        """
        self.drift(stepcoef)
        self.kick(stepcoef)
        self.forceDKD(self.gather(), stepcoef)
        self.kick(stepcoef)
        self.drift(stepcoef)


    @selftimer
    def step(self):
        """

        """
        ncoefs = len(self.coefs)
        for coef in self.coefs:
            self.stepDKD(ncoefs*coef)







########## end of file ##########
