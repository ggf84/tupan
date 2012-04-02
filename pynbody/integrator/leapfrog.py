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


    def get_min_tstep(self, old_tau):
        inv_tau = self.particles.set_tstep(self.particles, old_tau)
        max_inv_tau = 0.0
        for (key, value) in inv_tau.items():
            if value is not None:
                max_inv_tau = max(max_inv_tau, value.max())
        new_tau = float(self.eta / max_inv_tau**0.5)
        return new_tau


    @timings
    def drift(self, p, tau):
        """

        """
        self.time += tau
        for (key, obj) in p.items():
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


    def stepDKD(self, ip, jp, tau):
        """

        """
        self.drift(ip, 0.5 * tau)
        if jp: self.kick(jp, ip, tau)
        self.kick(ip, ip, tau)
        if jp: self.kick(ip, jp, tau)
        self.drift(ip, 0.5 * tau)



    def rstep(self, p, tau, update):
        if update: self.update_tstep(p, p, tau)
        slow, fast = self.split(tau, p)

        if fast: self.rstep(fast, tau/2, False)
        if slow: self.stepDKD(slow, fast, tau)
        if fast: self.rstep(fast, tau/2, True)


    @timings
    def step(self):
        """

        """
        self.tstep = self.get_min_tstep(0.5 * self.tstep)
        self.stepDKD(self.particles, [], self.tstep)

#        self.rstep(self.particles, 0.125, True)


    # Pickle-related methods

    def __getstate__(self):
        sdict = self.__dict__.copy()
        return sdict

    def __setstate__(self, sdict):
        self.__dict__.update(sdict)
        self.particles = self.particles.copy()


########## end of file ##########
