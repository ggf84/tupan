#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import logging
import numpy as np
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Hermite"]

logger = logging.getLogger(__name__)


@decallmethods(timings)
class Hermite(object):
    """

    """
    def __init__(self, eta, time, particles, **kwargs):
        self.eta = eta
        self.time = time
        self.particles = particles
        self.tstep = 0.0
        self.is_initialized = False

        self.pn_order = kwargs.pop("pn_order", 0)
        self.clight = kwargs.pop("clight", None)
        if self.pn_order > 0 and self.clight is None:
            raise TypeError("'clight' is not defined. Please set the input "
                            "argument 'clight' when using 'pn_order' != 0.")

        self.dumpper = kwargs.pop("dumpper", None)
        self.snap_freq = kwargs.pop("snap_freq", 0)
        if kwargs:
            msg = "{0}.__init__ received unexpected keyword arguments: {1}."
            raise TypeError(msg.format(self.__class__.__name__,", ".join(kwargs.keys())))


    def predict(self, ip, tau):
        """

        """
        self.prev_pos = {}
        self.prev_vel = {}
        self.prev_acc = {}
        self.prev_jerk = {}
        for (key, obj) in ip.items():
            if obj.n:
                self.prev_pos[key] = obj.pos.copy()
                self.prev_vel[key] = obj.vel.copy()
                self.prev_acc[key] = obj.acc.copy()
                self.prev_jerk[key] = obj.jerk.copy()

                obj.pos += obj.vel * tau + obj.acc * tau**2 / 2 + obj.jerk * tau**3 / 6
                obj.vel += obj.acc * tau + obj.jerk * tau**2 / 2


    def correct(self, ip, tau):
        alpha = 7.0 / 6.0
        beta = 6 * alpha - 5
        for (key, obj) in ip.items():
            if obj.n:
                obj.vel = (self.prev_vel[key] + (self.prev_acc[key] + obj.acc) * tau / 2
                                              + (self.prev_jerk[key] - obj.jerk) * tau * tau / 12)

                obj.pos = (self.prev_pos[key] + (self.prev_vel[key] + obj.vel) * tau / 2
                                              + alpha * (self.prev_acc[key] - obj.acc) * tau * tau / 10
                                              + beta * (self.prev_jerk[key] + obj.jerk) * tau * tau * tau / 120)


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
        prev_jerk = {}
        prev_pnacc = {}
        for (key, obj) in ip.items():
            if obj.n:
                if hasattr(obj, "acc"):
                    prev_acc[key] = obj.acc.copy()
                if hasattr(obj, "jerk"):
                    prev_jerk[key] = obj.jerk.copy()
                if self.pn_order > 0:
                    if hasattr(obj, "pnacc"):
                        prev_pnacc[key] = obj.pnacc.copy()

        ip.update_acc_jerk(jp)
        if self.pn_order > 0 and self.clight > 0:
            ip.update_pnacc(jp, self.pn_order, self.clight)

        for (key, obj) in ip.items():
            if obj.n:
                if hasattr(obj, "acc"):
                    obj.acc = 2 * obj.acc - prev_acc[key]
                if hasattr(obj, "jerk"):
                    obj.jerk = 2 * obj.jerk - prev_jerk[key]
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
#                if hasattr(obj, "jerk"):
#                    obj.pos -= obj.acc * tau * tau / 10
                if hasattr(obj, "evolve_velocity"):
                    obj.evolve_velocity(tau / 2)
#                if hasattr(obj, "jerk"):
#                    obj.vel += obj.jerk * tau * tau / 12

        self.forceDKD(ip, jp)

        for (key, obj) in ip.items():
            if obj.n:
#                if hasattr(obj, "jerk"):
#                    obj.vel -= obj.jerk * tau * tau / 12
                if hasattr(obj, "evolve_velocity"):
                    obj.evolve_velocity(tau / 2)
#                if hasattr(obj, "jerk"):
#                    obj.pos += obj.acc * tau * tau / 10
                if self.pn_order > 0:
                    if hasattr(obj, "evolve_velocity_correction_due_to_pnterms"):
                        obj.evolve_velocity_correction_due_to_pnterms(tau / 2)
                    if hasattr(obj, "evolve_energy_correction_due_to_pnterms"):
                        obj.evolve_energy_correction_due_to_pnterms(tau / 2)
                    if hasattr(obj, "evolve_angular_momentum_correction_due_to_pnterms"):
                        obj.evolve_angular_momentum_correction_due_to_pnterms(tau / 2)
                    if hasattr(obj, "evolve_linear_momentum_correction_due_to_pnterms"):
                        obj.evolve_linear_momentum_correction_due_to_pnterms(tau / 2)


    def dkd(self, p, tau):
        """

        """
        self.predict(p, tau)
        p.update_acc_jerk(p)

#        for (key, obj) in p.items():
#            if obj.n:
#                obj.acc = 2 * obj.acc - self.prev_acc[key]
#                obj.jerk = 2 * obj.jerk - self.prev_jerk[key]

        self.correct(p, tau)


#        self.drift(p, tau / 2)
#        self.kick(p, p, tau)
#        self.drift(p, tau / 2)

        return p


    def initialize(self, t_end):
        logger.info("Initializing integrator.")

        p = self.particles

        p.update_acc_jerk(p)
        if self.pn_order > 0:
            p.update_pnacc(p, self.pn_order, self.clight)
        p.update_timestep(p, self.eta)

        tau = self.eta
        p.set_dt_next(tau)
        self.tstep = tau

        self.snap_counter = self.snap_freq
        self.is_initialized = True


    def finalize(self, t_end):
        logger.info("Finalizing integrator.")

        p = self.particles
        tau = self.eta
        p.set_dt_next(tau)

        if self.dumpper:
            self.dumpper.dump(p)


    def step(self, t_end):
        """

        """
        if not self.is_initialized:
            self.initialize(t_end)

        tau = self.tstep

        p = self.particles
        p.set_dt_next(tau)

        if self.dumpper:
            self.snap_counter += 1
            if (self.snap_counter >= self.snap_freq):
                self.snap_counter -= self.snap_freq
                self.dumpper.dump(p)

        self.time += self.tstep / 2
        p = self.dkd(p, tau)
        self.time += self.tstep / 2

        p.set_dt_prev(tau)
        self.particles = p


########## end of file ##########
