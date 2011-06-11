#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np

from ggf84decor import selftimer


class LeapFrog(object):
    """

    """
    def __init__(self, time, eta, particles):
        self.time = time
        self.eta = eta
        self.particles = particles
        self.tstep = self.set_nexttstep(time)


    @selftimer
    def gather(self):
        return self.particles


    @selftimer
    def drift(self, dt):
        """

        """
        for (key, obj) in self.particles.iteritems():
            if obj:
                obj.drift(dt)


    @selftimer
    def kick(self, dt):
        """

        """
        for (key, obj) in self.particles.iteritems():
            if obj:
                obj.kick(dt)


    @selftimer
    def set_nexttstep(self, nexttime):
        mintsteps = []
        for (key, obj) in self.particles.iteritems():
            if obj:
                real_tstep = self.eta / obj.stepdens[:,1]
                mintsteps.append(np.min(real_tstep))
        power = int(np.log2(min(mintsteps)) - 1)
        nexttstep = 2.0**power
        while (nexttime+2*nexttstep)%(2*nexttstep) != 0:
            nexttstep /= 2
        return nexttstep


    @selftimer
    def forceDKD(self, all_particles, nexttime):
        """

        """
        for (key0, obj0) in self.particles.iteritems():
            if obj0:
                sum_acc = np.zeros_like(obj0.acc)
                sum_stepdens = np.zeros_like(obj0.stepdens[:,0])
                prev_acc = obj0.acc.copy()
                prev_stepdens = obj0.stepdens[:,0].copy()
                for (key1, obj1) in all_particles.iteritems():
                    if obj1:
                        obj0.set_acc(obj1)
                        sum_acc += obj0.acc
                        sum_stepdens += obj0.stepdens[:,0]
                obj0.acc[:] = 2*sum_acc - prev_acc
                obj0.stepdens[:,1] = (sum_stepdens**2) / (prev_stepdens)
                obj0.stepdens[:,0] = sum_stepdens

        return self.set_nexttstep(nexttime)


    def stepDKD(self):
        """

        """
        currtstep = self.tstep

        self.time += currtstep
        self.drift(currtstep)
        self.kick(currtstep)
        nexttstep = self.forceDKD(self.gather(), self.time+currtstep)
        self.kick(currtstep)
        self.drift(currtstep)
        self.time += currtstep

        self.tstep = nexttstep


    @selftimer
    def forceKDK(self, all_particles, nexttime):
        """

        """
        for (key0, obj0) in self.particles.iteritems():
            if obj0:
                sum_acc = np.zeros_like(obj0.acc)
                sum_stepdens = np.zeros_like(obj0.stepdens[:,0])
                prev_stepdens = obj0.stepdens[:,0].copy()
                for (key1, obj1) in all_particles.iteritems():
                    if obj1:
                        obj0.set_acc(obj1)
                        sum_acc += obj0.acc
                        sum_stepdens += obj0.stepdens[:,0]
                obj0.acc[:] = sum_acc
                obj0.stepdens[:,1] = (sum_stepdens**2) / prev_stepdens
                obj0.stepdens[:,0] = 0.5*(sum_stepdens + obj0.stepdens[:,1])

        return self.set_nexttstep(nexttime)


    def stepKDK(self):
        """

        """
        currtstep = self.tstep

        self.time += currtstep
        self.kick(currtstep)
        self.drift(currtstep)
        self.drift(currtstep)
        nexttstep = self.forceKDK(self.gather(), self.time+currtstep)
        self.kick(currtstep)
        self.time += currtstep

        self.tstep = nexttstep


    @selftimer
    def step(self):
        """

        """
        self.stepDKD()
#        self.stepKDK()


########## end of file ##########
