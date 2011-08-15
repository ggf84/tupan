#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np

from ggf84decor import selftimer


__all__ = ["LeapFrog"]


class LeapFrog(object):
    """

    """
    def __init__(self, eta, time, particles):
        self.eta = eta
        self.time = time
        particles.set_phi(particles)
        self.e0 = particles.get_total_energies()
        rhostep = particles.set_acc(particles)
        self.particles = particles


    @selftimer
    def gather(self):
        return self.particles


    @selftimer
    def drift(self, step_size):
        """

        """
        e = self.particles.get_total_energies()
        varstep = 0.5 / (e.kin - self.e0.tot)
#        varstep = 1.0
        tau = 0.5 * self.eta * step_size * varstep
        self.time += tau
        for (key, obj) in self.particles.iteritems():
            if obj:
                obj.drift(tau)


    @selftimer
    def kick(self, step_size):
        """

        """
        e = self.particles.get_total_energies()
        varstep = 0.5 / (-e.pot)
#        varstep = 1.0
        tau = 0.5 * self.eta * step_size * varstep
        for (key, obj) in self.particles.iteritems():
            if obj:
                obj.kick(tau)


    @selftimer
    def forceDKD(self, jparticles):
        """

        """
        rhostep = self.particles.set_acc(jparticles)
        self.particles.set_phi(jparticles)


    def stepDKD(self, step_size):
        """

        """
        self.drift(step_size)
        self.forceDKD(self.gather())
        self.kick(step_size)
        self.kick(step_size)
        self.drift(step_size)


    @selftimer
    def step(self):
        """

        """
        coefs = [1.0]

#        coefs = [1.3512071919596575,
#                -1.7024143839193150,
#                 1.3512071919596575]

#        coefs = [0.4144907717943757,
#                 0.4144907717943757,
#                -0.6579630871775028,
#                 0.4144907717943757,
#                 0.4144907717943757]

#        coefs = [0.3221375960817983,
#                 0.5413165481700432,
#                -0.7269082885036829,
#                 0.5413165481700432,
#                 0.3221375960817983]

#        coefs = [0.7845136104775573,
#                 0.23557321335935813,
#                -1.177679984178871,
#                 1.3151863206839112,
#                -1.177679984178871,
#                 0.23557321335935813,
#                 0.7845136104775573]

#        coefs = [0.1867,
#                 0.5554970237124784,
#                 0.12946694891347535,
#                -0.8432656233877346,
#                 0.9432033015235617,
#                -0.8432656233877346,
#                 0.12946694891347535,
#                 0.5554970237124784,
#                 0.1867]


        for coef in coefs:
            self.stepDKD(len(coefs)*coef)







########## end of file ##########
