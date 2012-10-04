#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import logging
import math
import numpy as np
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Hermite", "AdaptHermite"]

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
            raise TypeError(msg.format(type(self).__name__,", ".join(kwargs.keys())))


    def predict(self, ip, tau):
        """

        """
        ip.prev_pos = ip.pos.copy()
        ip.prev_vel = ip.vel.copy()
        ip.prev_acc = ip.acc.copy()
        ip.prev_jerk = ip.jerk.copy()

        ip.pos += tau * (ip.vel + (tau/2) * (ip.acc + (tau/3) * ip.jerk))
        ip.vel += tau * (ip.acc + (tau/2) * ip.jerk)


    def correct(self, ip, tau):
        ip.vel[:] = (ip.prev_vel + tau * ((ip.prev_acc + ip.acc)/2
                                 + tau * (ip.prev_jerk - ip.jerk)/12))
        ip.pos[:] = (ip.prev_pos + tau * ((ip.prev_vel + ip.vel)/2
                                 + tau * (ip.prev_acc - ip.acc)/12))


    def pec(self, n, p, tau):
        """

        """
        self.predict(p, tau)
        for i in range(n):
            p.update_acc_jerk(p)
            self.correct(p, tau)

        p.tstep[:] = tau
        p.time += tau
        p.nstep += 1
        return p


    def get_base_tstep(self, t_end):
        self.tstep = self.eta
        if abs(self.time + self.tstep) > t_end:
            self.tstep = math.copysign(t_end - abs(self.time), self.eta)
        return self.tstep


    def initialize(self, t_end):
        logger.info("Initializing '%s' integrator.", type(self).__name__.lower())

        p = self.particles

        p.update_acc_jerk(p)
        if self.pn_order > 0: p.update_pnacc(p, self.pn_order, self.clight)

        if self.dumpper:
            self.snap_number = 0
            self.dumpper.setup()
            self.dumpper.dump_snapshot(p, self.snap_number)

        self.is_initialized = True


    def finalize(self, t_end):
        logger.info("Finalizing '%s' integrator.", type(self).__name__.lower())

        p = self.particles
        tau = self.get_base_tstep(t_end)
        p.tstep[:] = tau

        if self.reporter:
            self.reporter.report(self.time, p)


    def evolve_step(self, t_end):
        """

        """
        if not self.is_initialized:
            self.initialize(t_end)

        p = self.particles
        tau = self.get_base_tstep(t_end)
        p.tstep[:] = tau

        if self.reporter:
            self.reporter.report(self.time, p)
        worldline = self.dumpper

        p = self.pec(2, p, tau)
        self.time += tau

        if self.dumpper:
            pp = p[p.nstep % self.dump_freq == 0]
            if pp.n:
                self.snap_number += 1
                self.dumpper.dump_snapshot(pp, self.snap_number)

        self.particles = p



@decallmethods(timings)
class AdaptHermite(Hermite):
    """

    """
    def __init__(self, eta, time, particles, **kwargs):
        super(AdaptHermite, self).__init__(eta, time, particles, **kwargs)


    def get_min_block_tstep(self, p):
        min_tstep = p.min_tstep()

        power = int(np.log2(min_tstep) - 1)
        min_block_tstep = 2.0**power

        next_time = self.time + min_block_tstep
        if next_time % min_block_tstep != 0:
            min_block_tstep /= 2

        return math.copysign(min_block_tstep, self.eta)


    def get_base_tstep(self, t_end):
        p = self.particles
        p.update_tstep(p, self.eta)
        tau = self.get_min_block_tstep(p)
        self.tstep = tau
        if abs(self.time + self.tstep) > t_end:
            self.tstep = math.copysign(t_end - abs(self.time), self.eta)
        return self.tstep


########## end of file ##########
