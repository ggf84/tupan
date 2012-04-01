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
        particles.set_acc(particles, 0.0)
        self.particles = particles


    @timings
    def gather(self):
        return self.particles


    def get_min_tstep(self, old_tstep):
        inv_tstep = self.particles.set_tstep(self.gather(), old_tstep)
        max_inv_tstep = 0.0
        for (key, value) in inv_tstep.items():
            if value is not None:
                max_inv_tstep = max(max_inv_tstep, value.max())
        new_tstep = float(self.eta / max_inv_tstep**0.5)
        return new_tstep


    @timings
    def drift(self, tstep):
        """

        """
        self.time += tstep
        for (key, obj) in self.particles.iteritems():
            if hasattr(obj, "evolve_pos"):
                obj.evolve_pos(tstep)
            if hasattr(obj, "evolve_com_pos_jump"):
                obj.evolve_com_pos_jump(tstep)


    @timings
    def forceDKD(self, jparticles):
        """

        """
        prev_acc = {}
        prev_pnacc = {}
        for (key, obj) in self.particles.items():
            if hasattr(obj, "acc"):
                prev_acc[key] = obj.acc.copy()
            if hasattr(obj, "pnacc"):
                prev_pnacc[key] = obj.pnacc.copy()

        self.particles.set_acc(jparticles, 0.0)

        for (key, obj) in self.particles.items():
            if hasattr(obj, "acc"):
                obj.acc[:] = 2 * obj.acc - prev_acc[key]
            if hasattr(obj, "pnacc"):
                obj.pnacc[:] = 2 * obj.pnacc - prev_pnacc[key]


    @timings
    def kick(self, tstep):
        """

        """
        for (key, obj) in self.particles.iteritems():
            if hasattr(obj, "pnacc"):
                external_force = -(obj.mass * obj.pnacc.T).T
                if hasattr(obj, "evolve_com_vel_jump"):
                    obj.evolve_com_vel_jump(0.5 * tstep, external_force)
                if hasattr(obj, "evolve_linmom_jump"):
                    obj.evolve_linmom_jump(0.5 * tstep, external_force)
                if hasattr(obj, "evolve_angmom_jump"):
                    obj.evolve_angmom_jump(0.5 * tstep, external_force)
                if hasattr(obj, "evolve_energy_jump"):
                    obj.evolve_energy_jump(0.5 * tstep, external_force)
            if hasattr(obj, "evolve_vel"):
                obj.evolve_vel(0.5 * tstep)

        self.forceDKD(self.gather())

        for (key, obj) in self.particles.iteritems():
            if hasattr(obj, "evolve_vel"):
                obj.evolve_vel(0.5 * tstep)
            if hasattr(obj, "pnacc"):
                external_force = -(obj.mass * obj.pnacc.T).T
                if hasattr(obj, "evolve_com_vel_jump"):
                    obj.evolve_com_vel_jump(0.5 * tstep, external_force)
                if hasattr(obj, "evolve_linmom_jump"):
                    obj.evolve_linmom_jump(0.5 * tstep, external_force)
                if hasattr(obj, "evolve_angmom_jump"):
                    obj.evolve_angmom_jump(0.5 * tstep, external_force)
                if hasattr(obj, "evolve_energy_jump"):
                    obj.evolve_energy_jump(0.5 * tstep, external_force)


    def stepDKD(self, tstep):
        """

        """
        self.drift(0.5 * tstep)
        self.kick(tstep)
        self.drift(0.5 * tstep)


    @timings
    def step(self):
        """

        """
        self.tstep = self.get_min_tstep(0.5 * self.tstep)
        self.stepDKD(self.tstep)


    # Pickle-related methods

    def __getstate__(self):
        sdict = self.__dict__.copy()
        return sdict

    def __setstate__(self, sdict):
        self.__dict__.update(sdict)
        self.particles = self.particles.copy()


########## end of file ##########
