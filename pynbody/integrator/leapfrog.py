#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
from ..lib.utils.timing import timings


__all__ = ["LeapFrog"]


class LeapFrog(object):
    """

    """
    def __init__(self, eta, time, particles):
        self.eta = eta
        self.time = time
        self.tstep = 0.0
        particles.set_acc(particles)
        self.particles = particles


    def get_min_tstep(self):
        min_tstep = 1.0
        for (key, obj) in self.particles.items():
            if obj:
                min_tstep = min(min_tstep, obj.tstep.min())
        return min_tstep


    @timings
    def drift(self, ip, tau):
        """

        """
        for (key, obj) in ip.items():
            if hasattr(obj, "evolve_pos"):
                obj.evolve_pos(tau)
            if hasattr(obj, "evolve_com_pos_jump"):
                obj.evolve_com_pos_jump(tau)


    @timings
    def forceDKD(self, ip, jp):
        """

        """
        prev_acc = {}
        prev_pnacc = {}
        for (key, obj) in ip.items():
            if hasattr(obj, "acc"):
                prev_acc[key] = obj.acc.copy()
            if hasattr(obj, "pnacc"):
                prev_pnacc[key] = obj.pnacc.copy()

        ip.set_acc(jp)

        for (key, obj) in ip.items():
            if hasattr(obj, "acc"):
                obj.acc[:] = 2 * obj.acc - prev_acc[key]
            if hasattr(obj, "pnacc"):
                obj.pnacc[:] = 2 * obj.pnacc - prev_pnacc[key]


    @timings
    def kick(self, ip, jp, tau):
        """

        """
        for (key, obj) in ip.iteritems():
            if hasattr(obj, "pnacc"):
                external_force = -(obj.mass * obj.pnacc.T).T
                if hasattr(obj, "evolve_com_vel_jump"):
                    obj.evolve_com_vel_jump(0.5 * tau, external_force)
                if hasattr(obj, "evolve_linmom_jump"):
                    obj.evolve_linmom_jump(0.5 * tau, external_force)
                if hasattr(obj, "evolve_angmom_jump"):
                    obj.evolve_angmom_jump(0.5 * tau, external_force)
                if hasattr(obj, "evolve_energy_jump"):
                    obj.evolve_energy_jump(0.5 * tau, external_force)
            if hasattr(obj, "evolve_vel"):
                obj.evolve_vel(0.5 * tau)

        self.forceDKD(ip, jp)

        for (key, obj) in ip.iteritems():
            if hasattr(obj, "evolve_vel"):
                obj.evolve_vel(0.5 * tau)
            if hasattr(obj, "pnacc"):
                external_force = -(obj.mass * obj.pnacc.T).T
                if hasattr(obj, "evolve_com_vel_jump"):
                    obj.evolve_com_vel_jump(0.5 * tau, external_force)
                if hasattr(obj, "evolve_linmom_jump"):
                    obj.evolve_linmom_jump(0.5 * tau, external_force)
                if hasattr(obj, "evolve_angmom_jump"):
                    obj.evolve_angmom_jump(0.5 * tau, external_force)
                if hasattr(obj, "evolve_energy_jump"):
                    obj.evolve_energy_jump(0.5 * tau, external_force)


    def stepDKD(self, slow, fast, tau):
        """

        """
        if not fast.any(): self.time += 0.5 * tau
        self.drift(slow, 0.5 * tau)
        if fast.any(): self.kick(fast, slow, tau)
        self.kick(slow, slow, tau)
        if fast.any(): self.kick(slow, fast, tau)
        self.drift(slow, 0.5 * tau)
        if not fast.any(): self.time += 0.5 * tau

    def update_tstep(self, ip, jp, tau):
        ip.set_tstep(jp, self.eta, tau)

    def split_by(self, tau, p):
        slow = p.__class__()
        fast = p.__class__()
        for (key, obj) in p.items():
            if obj:
                is_less = obj.tstep < tau
                is_greater = ~is_less
                slow[key] = obj[np.where(is_greater)]   # XXX: known bug: numpy fancy indexing returns a copy
                fast[key] = obj[np.where(is_less)]      #      but which we want is a view.

        return slow, fast

    def rstep(self, p, tau, update):
        if update: self.update_tstep(p, p, tau/2)
        slow, fast = self.split_by(tau, p)

        if fast.any(): self.rstep(fast, tau/2, False)
        if slow.any(): self.stepDKD(slow, fast, tau)
        if fast.any(): self.rstep(fast, tau/2, True)


    @timings
    def step(self):
        """

        """
        old_tstep = 0.5 * self.tstep
        self.particles.set_tstep(self.particles, self.eta, old_tstep)
        self.tstep = self.get_min_tstep()
        self.stepDKD(self.particles, np.array([]), self.tstep)

#        tau = 0.015625
#        self.rstep(self.particles, tau, True)


    # Pickle-related methods

    def __getstate__(self):
        sdict = self.__dict__.copy()
        return sdict

    def __setstate__(self, sdict):
        self.__dict__.update(sdict)
        self.particles = self.particles.copy()


########## end of file ##########
