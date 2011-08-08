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
    def __init__(self, time, eta, particles):
        self.time = time
        self.eta = eta
        rhostep = particles.set_acc(particles)
        self.particles = particles

        self.rho = [{}, {}, {}, {}]
        for i in range(4):
            for (key, obj) in rhostep.iteritems():
                if obj is not None:
                    self.rho[i][key] = obj.copy()

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
                real_tstep = self.eta / self.rho[3][key]
                mintsteps.append(np.min(real_tstep))
        power = int(np.log2(min(mintsteps)) - 1)
        nexttstep = 2.0**power
        while (nexttime+2*nexttstep)%(2*nexttstep) != 0:
            nexttstep /= 2
        return nexttstep
#        return min(mintsteps)


    @selftimer
    def forceDKD(self, other_particles, nexttime):
        """

        """
        for (key, obj) in self.particles.iteritems():
            if obj:
                prev_acc = obj.acc.copy()
                rhostep = obj.set_acc(other_particles)
                obj.acc += obj.acc - prev_acc

                self.rho[0][key][:] = (self.rho[0][key] * self.rho[3][key])**0.5
                self.rho[1][key][:] = self.rho[2][key]
                self.rho[2][key][:] = rhostep
                self.rho[3][key][:] = self.rho[0][key] * (self.rho[2][key] / self.rho[1][key])

        return self.set_nexttstep(nexttime)



#    @selftimer
#    def forceDKD(self, all_particles, nexttime):
#        """

#        """
#        for (key0, obj0) in self.particles.iteritems():
#            if obj0:
##                self.rho[0][key0][:] = self.rho[1][key0].copy()
###                self.rho[1][key0][:] = (self.rho[1][key0] + self.rho[2][key0])/2
##                self.rho[1][key0][:] = self.rho[2][key0].copy()
##                self.rho[2][key0][:] = self.rho[3][key0].copy()

#                sum_acc = np.zeros_like(obj0.acc)
#                sum_stepdens = np.zeros_like(obj0.stepdens[:,0])
#                prev_acc = obj0.acc.copy()
#                prev_stepdens = obj0.stepdens[:,0].copy()
#                for (key1, obj1) in all_particles.iteritems():
#                    if obj1:
#                        obj0.set_acc(obj1)
#                        sum_acc += obj0.acc
#                        sum_stepdens += obj0.stepdens[:,0]
#                obj0.acc[:] = 2*sum_acc - prev_acc
##                obj0.stepdens[:,1] = (sum_stepdens**2) / (prev_stepdens)
##                obj0.stepdens[:,0] = sum_stepdens

##                obj0.stepdens[:,1] *= (sum_stepdens / prev_stepdens)**0.5

#                obj0.stepdens[:,1] = 0.5*(sum_stepdens + prev_stepdens) * (sum_stepdens / prev_stepdens)**2.5

### XXX: 3788::5256::7353
##                self.rho[0][key0][:] = (self.rho[1][key0] + self.rho[2][key0])/2
##                self.rho[1][key0][:] = self.rho[2][key0].copy()
##                self.rho[2][key0][:] = sum_stepdens.copy()
##                self.rho[3][key0][:] = (self.rho[1][key0] * self.rho[2][key0]) / self.rho[0][key0]


### XXX: 3668::5661::7514
##                self.rho[0][key0][:] = (self.rho[1][key0]*self.rho[2][key0])**0.5
##                self.rho[1][key0][:] = self.rho[2][key0].copy()
##                self.rho[2][key0][:] = sum_stepdens.copy()
##                self.rho[3][key0][:] = (self.rho[1][key0] * self.rho[2][key0]) / self.rho[0][key0]


### XXX: 3670::5747::7098
##                self.rho[0][key0][:] = (self.rho[1][key0]*self.rho[2][key0])**0.5
##                self.rho[1][key0][:] = self.rho[2][key0].copy()
##                self.rho[2][key0][:] = sum_stepdens.copy()
##                self.rho[3][key0][:] = (((self.rho[1][key0] + self.rho[2][key0])/2)**2) / self.rho[0][key0]


### XXX: 3998::5637::7132
##                self.rho[0][key0][:] = (self.rho[1][key0] + self.rho[2][key0])/2
##                self.rho[1][key0][:] = self.rho[2][key0].copy()
##                self.rho[2][key0][:] = sum_stepdens.copy()
##                self.rho[3][key0][:] = (((self.rho[1][key0] + self.rho[2][key0])/2)**2) / self.rho[0][key0]



## XXX: 2925:: ::
#                self.rho[0][key0][:] = (self.rho[0][key0]*self.rho[3][key0])**0.5
#                self.rho[1][key0][:] = self.rho[2][key0].copy()
#                self.rho[2][key0][:] = sum_stepdens.copy()
#                self.rho[3][key0][:] = self.rho[0][key0] * (self.rho[2][key0] / self.rho[1][key0])


#        return self.set_nexttstep(nexttime)


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
