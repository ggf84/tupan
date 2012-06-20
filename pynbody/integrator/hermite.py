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

                obj.pos += tau * (obj.vel + (tau/2) * (obj.acc + (tau/3) * obj.jerk))
                obj.vel += tau * (obj.acc + (tau/2) * obj.jerk)


    def correct(self, ip, tau):
        for (key, obj) in ip.items():
            if obj.n:
                obj.vel = (self.prev_vel[key] + tau * ((self.prev_acc[key] + obj.acc)/2
                                              + tau * (self.prev_jerk[key] - obj.jerk)/12))
                obj.pos = (self.prev_pos[key] + tau * ((self.prev_vel[key] + obj.vel)/2
                                              + tau * (self.prev_acc[key] - obj.acc)/12))


    def pec(self, n, p, tau):
        """

        """
        self.predict(p, tau)
        for i in range(n):
            p.update_acc_jerk(p)
            self.correct(p, tau)

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

        p.update_acc_jerk(p)
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

        p = self.pec(2, p, tau)
        self.time += tau

        self.particles = p


########## end of file ##########
