#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np


class LeapFrog(object):
    """

    """
    def __init__(self, time, eta, particles):
        self.time = time
        self.eta = eta
        self.particles = particles
        self.tstep = eta
        self.set_tstep()

    def set_tstep(self):
        tstep = []
        for (key, obj) in self.particles.iteritems():
            if obj:
                real_tstep = self.eta / obj.stepdens[:,1]
                tstep.append(np.min(real_tstep))
        power = int(np.log2(min(tstep)) - 1)
        self.tstep = 2.0**power



    def drift(self):
        """

        """
        for (key, obj) in self.particles.iteritems():
            if obj:
                obj.drift(self.tstep)


    def kick(self):
        """

        """
        for (key, obj) in self.particles.iteritems():
            if obj:
                obj.kick(self.tstep)


    def force(self, all_particles):
        """

        """
        for (key0, obj0) in self.particles.iteritems():
            if obj0:
                prev_acc = obj0.acc.copy()
                mid_acc = np.zeros_like(obj0.acc)
                prev_stepdens = obj0.stepdens[:,0].copy()
                mid_stepdens = np.zeros_like(obj0.stepdens[:,0])
                for (key1, obj1) in all_particles.iteritems():
                    if obj1:
                        obj0.set_acc(obj1)
                        mid_acc += obj0.acc
                        mid_stepdens += obj0.stepdens[:,0]
                obj0.acc[:] = 2*mid_acc - prev_acc
                obj0.stepdens[:,0] = (mid_stepdens**2) / prev_stepdens
                obj0.stepdens[:,1] = (obj0.stepdens[:,0]**2) / mid_stepdens


    def step(self):
        self.time += self.tstep
        self.drift()
        self.kick()
        self.force(self.particles)
        self.kick()
        self.drift()
        self.time += self.tstep

        self.set_tstep()



########## end of file ##########
