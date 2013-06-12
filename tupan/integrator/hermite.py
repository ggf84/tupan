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
    def __init__(self, eta, time, ps, **kwargs):
        """

        """
        super(Hermite, self).__init__(eta, time, ps, **kwargs)

    def get_base_tstep(self, t_end):
        """

        """
        self.tstep = self.eta
        if abs(self.time + self.tstep) > t_end:
            self.tstep = math.copysign(t_end - abs(self.time), self.eta)
        return self.tstep

    def initialize(self, t_end):
        """

        """
        logger.info(
            "Initializing '%s' integrator.",
            type(self).__name__.lower()
        )

        ps = self.ps
        (ps.ax, ps.ay, ps.az, ps.jx, ps.jy, ps.jz) = ps.get_acc_jerk(ps)

        if self.reporter:
            self.reporter.diagnostic_report(self.time, ps)
        if self.dumpper:
            self.dumpper.dump_worldline(ps)

        self.is_initialized = True

    def finalize(self, t_end):
        """

        """
        logger.info(
            "Finalizing '%s' integrator.",
            type(self).__name__.lower()
        )

    def predict(self, ps, tau):
        """

        """
        (ax, ay, az, jx, jy, jz) = ps.get_acc_jerk(ps)
        ps.prev_rx = ps.rx.copy()
        ps.prev_ry = ps.ry.copy()
        ps.prev_rz = ps.rz.copy()
        ps.prev_vx = ps.vx.copy()
        ps.prev_vy = ps.vy.copy()
        ps.prev_vz = ps.vz.copy()
        ps.prev_ax = ax.copy()
        ps.prev_ay = ay.copy()
        ps.prev_az = az.copy()
        ps.prev_jx = jx.copy()
        ps.prev_jy = jy.copy()
        ps.prev_jz = jz.copy()

        ps.rx += tau * (ps.vx + (tau/2) * (ax + (tau/3) * jx))
        ps.ry += tau * (ps.vy + (tau/2) * (ay + (tau/3) * jy))
        ps.rz += tau * (ps.vz + (tau/2) * (az + (tau/3) * jz))
        ps.vx += tau * (ax + (tau/2) * jx)
        ps.vy += tau * (ay + (tau/2) * jy)
        ps.vz += tau * (az + (tau/2) * jz)

    def ecorrect(self, ps, tau):
        """

        """
        (ps.ax, ps.ay, ps.az, ps.jx, ps.jy, ps.jz) = ps.get_acc_jerk(ps)
        ps.vx = (ps.prev_vx + tau * ((ps.prev_ax + ps.ax)/2
                            + tau * (ps.prev_jx - ps.jx)/12))
        ps.vy = (ps.prev_vy + tau * ((ps.prev_ay + ps.ay)/2
                            + tau * (ps.prev_jy - ps.jy)/12))
        ps.vz = (ps.prev_vz + tau * ((ps.prev_az + ps.az)/2
                            + tau * (ps.prev_jz - ps.jz)/12))
        ps.rx = (ps.prev_rx + tau * ((ps.prev_vx + ps.vx)/2
                            + tau * (ps.prev_ax - ps.ax)/12))
        ps.ry = (ps.prev_ry + tau * ((ps.prev_vy + ps.vy)/2
                            + tau * (ps.prev_ay - ps.ay)/12))
        ps.rz = (ps.prev_rz + tau * ((ps.prev_vz + ps.vz)/2
                            + tau * (ps.prev_az - ps.az)/12))

    def pec(self, n, ps, tau):
        """

        """
        self.predict(ps, tau)
        for i in range(n):
            self.ecorrect(ps, tau)
        return ps

    def do_step(self, ps, tau):
        """

        """
        ps = self.pec(1, ps, tau)

        ps.tstep = tau
        ps.time += tau
        ps.nstep += 1
        wp = ps[ps.nstep % self.dump_freq == 0]
        if wp.n:
            self.wl.append(wp.copy())
        return ps

    def evolve_step(self, t_end):
        """

        """
        if not self.is_initialized:
            self.initialize(t_end)

        ps = self.ps
        tau = self.get_base_tstep(t_end)
        self.wl = ps[:0]

        ps = self.do_step(ps, tau)

        self.time += tau
        self.ps = ps

        if self.reporter:
            self.reporter.diagnostic_report(self.time, ps)
        if self.dumpper:
            self.dumpper.dump_worldline(self.wl)


@decallmethods(timings)
class AdaptHermite(Hermite):
    """

    """
    def __init__(self, eta, time, ps, **kwargs):
        """

        """
        super(AdaptHermite, self).__init__(eta, time, ps, **kwargs)

    def get_min_block_tstep(self, ps):
        """

        """
        min_tstep = ps.min_tstep()

        power = int(np.log2(min_tstep) - 1)
        min_block_tstep = 2.0**power

        next_time = self.time + min_block_tstep
        if next_time % min_block_tstep != 0:
            min_block_tstep /= 2

        return math.copysign(min_block_tstep, self.eta)

    def get_base_tstep(self, t_end):
        """

        """
        ps = self.ps
        ps.update_tstep(ps, self.eta)
        tau = self.get_min_block_tstep(ps)
        self.tstep = tau
        if abs(self.time + self.tstep) > t_end:
            self.tstep = math.copysign(t_end - abs(self.time), self.eta)
        return self.tstep


########## end of file ##########
