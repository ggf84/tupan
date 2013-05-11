# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import logging
import math
import numpy as np
from ..integrator import Base
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Hermite", "AdaptHermite"]

logger = logging.getLogger(__name__)


@decallmethods(timings)
class Hermite(Base):
    """

    """
    def __init__(self, eta, time, particles, **kwargs):
        super(Hermite, self).__init__(eta, time, particles, **kwargs)

    def predict(self, ip, tau):
        """

        """
        (ax, ay, az, jx, jy, jz) = ip.get_acc_jerk(ip)
        ip.prev_rx = ip.rx.copy()
        ip.prev_ry = ip.ry.copy()
        ip.prev_rz = ip.rz.copy()
        ip.prev_vx = ip.vx.copy()
        ip.prev_vy = ip.vy.copy()
        ip.prev_vz = ip.vz.copy()
        ip.prev_ax = ax.copy()
        ip.prev_ay = ay.copy()
        ip.prev_az = az.copy()
        ip.prev_jx = jx.copy()
        ip.prev_jy = jy.copy()
        ip.prev_jz = jz.copy()

        ip.rx += tau * (ip.vx + (tau/2) * (ax + (tau/3) * jx))
        ip.ry += tau * (ip.vy + (tau/2) * (ay + (tau/3) * jy))
        ip.rz += tau * (ip.vz + (tau/2) * (az + (tau/3) * jz))
        ip.vx += tau * (ax + (tau/2) * jx)
        ip.vy += tau * (ay + (tau/2) * jy)
        ip.vz += tau * (az + (tau/2) * jz)

    def ecorrect(self, ip, tau):
        (ax, ay, az, jx, jy, jz) = ip.get_acc_jerk(ip)
        ip.vx = (ip.prev_vx + tau * ((ip.prev_ax + ax)/2
                            + tau * (ip.prev_jx - jx)/12))
        ip.vy = (ip.prev_vy + tau * ((ip.prev_ay + ay)/2
                            + tau * (ip.prev_jy - jy)/12))
        ip.vz = (ip.prev_vz + tau * ((ip.prev_az + az)/2
                            + tau * (ip.prev_jz - jz)/12))
        ip.rx = (ip.prev_rx + tau * ((ip.prev_vx + ip.vx)/2
                            + tau * (ip.prev_ax - ax)/12))
        ip.ry = (ip.prev_ry + tau * ((ip.prev_vy + ip.vy)/2
                            + tau * (ip.prev_ay - ay)/12))
        ip.rz = (ip.prev_rz + tau * ((ip.prev_vz + ip.vz)/2
                            + tau * (ip.prev_az - az)/12))

    def pec(self, n, p, tau):
        """

        """
        self.predict(p, tau)
        for i in range(n):
            self.ecorrect(p, tau)

        p.tstep = tau
        p.time += tau
        p.nstep += 1
        return p

    def get_base_tstep(self, t_end):
        self.tstep = self.eta
        if abs(self.time + self.tstep) > t_end:
            self.tstep = math.copysign(t_end - abs(self.time), self.eta)
        return self.tstep

    def initialize(self, t_end):
        logger.info(
            "Initializing '%s' integrator.",
            type(self).__name__.lower()
        )

        p = self.particles

        if self.dumpper:
            self.snap_number = 0
            self.dumpper.dump_snapshot(p, self.snap_number)

        self.is_initialized = True

    def finalize(self, t_end):
        logger.info(
            "Finalizing '%s' integrator.",
            type(self).__name__.lower()
        )

        p = self.particles
        tau = self.get_base_tstep(t_end)
        p.tstep = tau

        if self.reporter:
            self.reporter.report(self.time, p)

    def evolve_step(self, t_end):
        """

        """
        if not self.is_initialized:
            self.initialize(t_end)

        p = self.particles
        tau = self.get_base_tstep(t_end)
        p.tstep = tau

        if self.reporter:
            self.reporter.report(self.time, p)

        p = self.pec(1, p, tau)
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
