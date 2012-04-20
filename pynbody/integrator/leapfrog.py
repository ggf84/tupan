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
    def __init__(self, eta, time, particles, **kwargs):
        self.eta = eta
        self.time = time
        self.particles = particles
        self.tstep = 0.0
        self.is_initialized = False
        self.n2_sum = 0
        self.nkick = 0
        self.dumpper = kwargs.pop("dumpper", None)
        self.snap_freq = kwargs.pop("snap_freq")
        if kwargs:
            msg = "{0}.__init__ received unexpected keyword arguments: {1}."
            raise TypeError(msg.format(self.__class__.__name__,
                                       ", ".join(kwargs.keys())))


    @timings
    def init_for_integration(self, t_end):
        logger.info("Initializing for integration.")

        p = self.particles

        p.update_acc(p)
        p.update_pnacc(p)
        p.update_timestep(p, self.eta)

        tau = self.get_min_block_tstep(p, t_end)
        p.set_dt_next(tau)
        self.tstep = tau

        self.snap_counter = self.snap_freq
        self.is_initialized = True


    def get_min_block_tstep(self, p, t_end):
        min_tstep = p.min_dt_next()

        power = int(np.log2(min_tstep) - 1)
        min_block_tstep = 2.0**power

        if (self.time+min_block_tstep)%(min_block_tstep) != 0:
            min_block_tstep /= 2

#        tau = min_block_tstep if self.time+min_block_tstep < t_end else t_end-self.time
#        return tau
        return min_block_tstep


    @timings
    def drift(self, ip, tau):
        """

        """
        for (key, obj) in ip.items():
            if obj:
                if hasattr(obj, "evolve_current_time"):
                    obj.evolve_current_time(tau)
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
        self.nkick += 1
        ntot = self.particles.get_nbody()
        if ni == ntot and nj == ntot:
            print(ni, nj, self.n2_sum)
#        print(self.n2_sum, self.nkick, self.n2_sum/self.nkick)


    @timings
    def dkd(self, p, tau):
        """

        """
        self.drift(p, tau / 2)
        self.kick(p, p, tau)
        self.drift(p, tau / 2)


    @timings
    def step(self, t_end):
        """

        """
        if not self.is_initialized:
            self.init_for_integration(t_end)

        tau = self.tstep
        p = self.particles

        tau = self.get_min_block_tstep(p, t_end)
        p.set_dt_next(tau)
        self.tstep = tau

        if self.dumpper:
            self.snap_counter += 1
            if (self.snap_counter >= self.snap_freq):
                self.snap_counter -= self.snap_freq
                self.dumpper.dump(p)

        self.time += self.tstep / 2
        self.dkd(p, tau)
        self.time += self.tstep / 2

        p.set_dt_prev(tau)
        p.update_timestep(p, self.eta)


########## end of file ##########
