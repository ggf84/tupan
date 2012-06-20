#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import logging
import numpy as np
from ..lib.utils.timing import decallmethods, timings


__all__ = ["LeapFrog", "AdaptLF"]

logger = logging.getLogger(__name__)


class Base(object):
    """

    """
    def __init__(self, eta, time, particles, **kwargs):
        self.eta = eta
        self.time = time
        self.particles = particles
        self.is_initialized = False

        self.pn_order = kwargs.pop("pn_order", 0)
        self.clight = kwargs.pop("clight", None)
        if self.pn_order > 0 and self.clight is None:
            raise TypeError("'clight' is not defined. Please set the speed of "
                            "light argument 'clight' when using 'pn_order' > 0.")

        self.reporter = kwargs.pop("reporter", None)
        self.dumpper = kwargs.pop("dumpper", None)
        self.dump_freq = kwargs.pop("dump_freq", 1)
        if kwargs:
            msg = "{0}.__init__ received unexpected keyword arguments: {1}."
            raise TypeError(msg.format(self.__class__.__name__,", ".join(kwargs.keys())))


    def drift(self, ip, tau):
        """

        """
        for (key, obj) in ip.items():
            if obj.n:
                if hasattr(obj, "evolve_current_time"):
                    obj.evolve_current_time(tau)
                if hasattr(obj, "evolve_position"):
                    obj.evolve_position(tau)
                if self.pn_order > 0:
                    if hasattr(obj, "evolve_center_of_mass_position_correction_due_to_pnterms"):
                        obj.evolve_center_of_mass_position_correction_due_to_pnterms(tau)


    def forceDKD(self, ip, jp):
        """

        """
        prev_acc = {}
        prev_pnacc = {}
        for (key, obj) in ip.items():
            if obj.n:
                if hasattr(obj, "acc"):
                    prev_acc[key] = obj.acc.copy()
                if self.pn_order > 0:
                    if hasattr(obj, "pnacc"):
                        prev_pnacc[key] = obj.pnacc.copy()

        ip.update_acc(jp)
        if self.pn_order > 0 and self.clight > 0:
            ip.update_pnacc(jp, self.pn_order, self.clight)

        for (key, obj) in ip.items():
            if obj.n:
                if hasattr(obj, "acc"):
                    obj.acc = 2 * obj.acc - prev_acc[key]
                if self.pn_order > 0:
                    if hasattr(obj, "pnacc"):
                        obj.pnacc = 2 * obj.pnacc - prev_pnacc[key]


    def kick(self, ip, jp, tau):
        """

        """
        for (key, obj) in ip.items():
            if obj.n:
                if self.pn_order > 0:
                    if hasattr(obj, "evolve_linear_momentum_correction_due_to_pnterms"):
                        obj.evolve_linear_momentum_correction_due_to_pnterms(tau / 2)
                    if hasattr(obj, "evolve_angular_momentum_correction_due_to_pnterms"):
                        obj.evolve_angular_momentum_correction_due_to_pnterms(tau / 2)
                    if hasattr(obj, "evolve_energy_correction_due_to_pnterms"):
                        obj.evolve_energy_correction_due_to_pnterms(tau / 2)
                    if hasattr(obj, "evolve_velocity_correction_due_to_pnterms"):
                        obj.evolve_velocity_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_velocity"):
                    obj.evolve_velocity(tau / 2)

        self.forceDKD(ip, jp)

        for (key, obj) in ip.items():
            if obj.n:
                if hasattr(obj, "evolve_velocity"):
                    obj.evolve_velocity(tau / 2)
                if self.pn_order > 0:
                    if hasattr(obj, "evolve_velocity_correction_due_to_pnterms"):
                        obj.evolve_velocity_correction_due_to_pnterms(tau / 2)
                    if hasattr(obj, "evolve_energy_correction_due_to_pnterms"):
                        obj.evolve_energy_correction_due_to_pnterms(tau / 2)
                    if hasattr(obj, "evolve_angular_momentum_correction_due_to_pnterms"):
                        obj.evolve_angular_momentum_correction_due_to_pnterms(tau / 2)
                    if hasattr(obj, "evolve_linear_momentum_correction_due_to_pnterms"):
                        obj.evolve_linear_momentum_correction_due_to_pnterms(tau / 2)



@decallmethods(timings)
class LeapFrog(Base):
    """

    """
    def __init__(self, eta, time, particles, **kwargs):
        super(LeapFrog, self).__init__(eta, time, particles, **kwargs)


    def dkd(self, p, tau):
        """

        """
        self.drift(p, tau / 2)
        self.kick(p, p, tau)
        self.drift(p, tau / 2)

        p.set_dt_prev(tau)
        p.update_nstep()
        return p


    def get_base_tstep(self, t_end):
        tau = self.eta
        self.tstep = tau if self.time + tau <= t_end else t_end - self.time
        return self.tstep


    def initialize(self, t_end):
        logger.info("Initializing '%s' integrator.", self.__class__.__name__.lower())

        p = self.particles

        p.update_acc(p)
        if self.pn_order > 0:
            p.update_pnacc(p, self.pn_order, self.clight)

        self.is_initialized = True


    def finalize(self, t_end):
        logger.info("Finalizing '%s' integrator.", self.__class__.__name__.lower())

        p = self.particles
        tau = self.get_base_tstep(t_end)
        p.set_dt_next(tau)

        if self.reporter:
            self.reporter.report(self.time, p)
        if self.dumpper:
            self.dumpper.dump(p)


    def evolve_step(self, t_end):
        """

        """
        if not self.is_initialized:
            self.initialize(t_end)

        p = self.particles
        tau = self.get_base_tstep(t_end)
        p.set_dt_next(tau)

        if self.reporter:
            self.reporter.report(self.time, p)
        if self.dumpper:
            self.dumpper.dump(p.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        p = self.dkd(p, tau)
        self.time += tau

        self.particles = p



@decallmethods(timings)
class AdaptLF(LeapFrog):
    """

    """
    def __init__(self, eta, time, particles, **kwargs):
        super(AdaptLF, self).__init__(eta, time, particles, **kwargs)


    def get_min_block_tstep(self, p):
        min_tstep = p.min_dt_next()

        power = int(np.log2(min_tstep) - 1)
        min_block_tstep = 2.0**power

        next_time = self.time + min_block_tstep
        if next_time % min_block_tstep != 0:
            min_block_tstep /= 2

        return min_block_tstep


    def get_base_tstep(self, t_end):
        p = self.particles
        p.update_tstep(p, self.eta)
        tau = self.get_min_block_tstep(p)
        self.tstep = tau if self.time + tau <= t_end else t_end - self.time
        return self.tstep


########## end of file ##########
