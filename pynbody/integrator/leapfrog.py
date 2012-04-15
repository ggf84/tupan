#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import logging
import numpy as np
from ..lib.utils.timing import timings


__all__ = ["LeapFrog"]

logger = logging.getLogger(__name__)


class LeapFrog(object):
    """

    """
    def __init__(self, eta, time, particles):
        self.eta = eta
        self.time = time
        particles.update_acctstep(particles, eta)
        particles.update_pnacc(particles)
        self.particles = particles
        self.tstep = self.get_min_block_tstep()
        self.n2_sum = 0


    def get_min_block_tstep(self):
        min_tstep = 1.0
        for (key, obj) in self.particles.items():
            if obj:
                min_tstep = min(min_tstep, obj.tstep.min())

        power = int(np.log2(min_tstep) - 1)
        min_tstep = 2.0**power

        if (self.time+min_tstep)%(min_tstep) != 0:
            min_tstep /= 2

        return min_tstep


    @timings
    def drift(self, ip, tau):
        """

        """
        for (key, obj) in ip.items():
            if obj:
                if hasattr(obj, "evolve_position"):
                    obj.evolve_position(tau)
                if hasattr(obj, "evolve_center_of_mass_position_correction_due_to_pnterms"):
                    obj.evolve_center_of_mass_position_correction_due_to_pnterms(tau)


    @timings
    def forceDKD(self, ip, jp):
        """

        """
        prev_acc = {}
        prev_pnacc = {}
        for (key, obj) in ip.items():
            if obj:
                if hasattr(obj, "acc"):
                    prev_acc[key] = obj.acc.copy()
                if hasattr(obj, "pnacc"):
                    prev_pnacc[key] = obj.pnacc.copy()

        ip.update_acc(jp)
        ip.update_pnacc(jp)

        for (key, obj) in ip.items():
            if obj:
                if hasattr(obj, "acc"):
                    obj.acc = 2 * obj.acc - prev_acc[key]
                if hasattr(obj, "pnacc"):
                    obj.pnacc = 2 * obj.pnacc - prev_pnacc[key]


    @timings
    def kick(self, ip, jp, tau):
        """

        """
        for (key, obj) in ip.items():
            if obj:
                if hasattr(obj, "evolve_linear_momentum_correction_due_to_pnterms"):
                    obj.evolve_linear_momentum_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_angular_momentum_correction_due_to_pnterms"):
                    obj.evolve_angular_momentum_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_center_of_mass_velocity_correction_due_to_pnterms"):
                    obj.evolve_center_of_mass_velocity_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_energy_correction_due_to_pnterms"):
                    obj.evolve_energy_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_velocity_correction_due_to_pnterms"):
                    obj.evolve_velocity_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_velocity"):
                    obj.evolve_velocity(tau / 2)

        self.forceDKD(ip, jp)

        for (key, obj) in ip.items():
            if obj:
                if hasattr(obj, "evolve_velocity"):
                    obj.evolve_velocity(tau / 2)
                if hasattr(obj, "evolve_velocity_correction_due_to_pnterms"):
                    obj.evolve_velocity_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_energy_correction_due_to_pnterms"):
                    obj.evolve_energy_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_center_of_mass_velocity_correction_due_to_pnterms"):
                    obj.evolve_center_of_mass_velocity_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_angular_momentum_correction_due_to_pnterms"):
                    obj.evolve_angular_momentum_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_linear_momentum_correction_due_to_pnterms"):
                    obj.evolve_linear_momentum_correction_due_to_pnterms(tau / 2)

        ni = ip.get_nbody()
        nj = jp.get_nbody()
        self.n2_sum += ni*nj
        ntot = self.particles.get_nbody()
#        if ni == ntot and nj == ntot:
#            print(ni, nj, self.n2_sum)


    @timings
    def dkd(self, p, tau):
        """

        """
        self.drift(p, tau / 2)
        self.kick(p, p, tau)
        self.drift(p, tau / 2)


    @timings
    def step(self):
        """

        """
        self.particles.update_tstep(self.particles, self.eta)
        self.tstep = self.get_min_block_tstep()

        self.time += self.tstep / 2
        self.dkd(self.particles, self.tstep)
        self.time += self.tstep / 2


########## end of file ##########
