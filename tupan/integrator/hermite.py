# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import logging
from ..integrator import Base
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Hermite"]

logger = logging.getLogger(__name__)


@decallmethods(timings)
class Hermite(Base):
    """

    """
    PROVIDED_METHODS = ['hermite4', 'ahermite4',
                        # TBD
                        # 'hermite6', 'ahermite6',
                        # 'hermite8', 'ahermite8',
                        ]

    def __init__(self, eta, time, ps, method, **kwargs):
        """

        """
        super(Hermite, self).__init__(eta, time, ps, **kwargs)
        self.method = method

    def initialize(self, t_end):
        """

        """
        logger.info("Initializing '%s' integrator.",
                    self.method)

        ps = self.ps
        (ps.ax, ps.ay, ps.az, ps.jx, ps.jy, ps.jz) = ps.get_acc_jerk(ps)

        if self.reporter:
            self.reporter.diagnostic_report(ps)
        if self.dumpper:
            self.dumpper.dump_worldline(ps)
        if self.viewer:
            self.viewer.show_event(ps)

        self.is_initialized = True

    def finalize(self, t_end):
        """

        """
        logger.info("Finalizing '%s' integrator.",
                    self.method)

        ps = self.ps

        if self.viewer:
            self.viewer.show_event(ps)
            self.viewer.enter_main_loop()

    def get_hermite_tstep(self, ps, eta, tau):
        """

        """
        ps.tstep[:], _ = ps.get_tstep(ps, eta)
        min_bts = self.get_min_block_tstep(ps, tau)
        return min_bts

    def predict(self, ps, tau):
        """

        """
        ps.rx0 = ps.rx.copy()
        ps.ry0 = ps.ry.copy()
        ps.rz0 = ps.rz.copy()
        ps.vx0 = ps.vx.copy()
        ps.vy0 = ps.vy.copy()
        ps.vz0 = ps.vz.copy()
        ps.ax0 = ps.ax.copy()
        ps.ay0 = ps.ay.copy()
        ps.az0 = ps.az.copy()
        ps.jx0 = ps.jx.copy()
        ps.jy0 = ps.jy.copy()
        ps.jz0 = ps.jz.copy()

        ps.rx += (ps.vx + (ps.ax + ps.jx * tau/3) * tau/2) * tau
        ps.ry += (ps.vy + (ps.ay + ps.jy * tau/3) * tau/2) * tau
        ps.rz += (ps.vz + (ps.az + ps.jz * tau/3) * tau/2) * tau
        ps.vx += (ps.ax + ps.jx * tau/2) * tau
        ps.vy += (ps.ay + ps.jy * tau/2) * tau
        ps.vz += (ps.az + ps.jz * tau/2) * tau

    def ecorrect(self, ps, tau):
        """

        """
        (ps.ax, ps.ay, ps.az, ps.jx, ps.jy, ps.jz) = ps.get_acc_jerk(ps)
        ps.vx[:] = (ps.vx0
                    + ((ps.ax0 + ps.ax)
                    + (ps.jx0 - ps.jx) * tau/6) * tau/2)
        ps.vy[:] = (ps.vy0
                    + ((ps.ay0 + ps.ay)
                    + (ps.jy0 - ps.jy) * tau/6) * tau/2)
        ps.vz[:] = (ps.vz0
                    + ((ps.az0 + ps.az)
                    + (ps.jz0 - ps.jz) * tau/6) * tau/2)
        ps.rx[:] = (ps.rx0
                    + ((ps.vx0 + ps.vx)
                    + (ps.ax0 - ps.ax) * tau/6) * tau/2)
        ps.ry[:] = (ps.ry0
                    + ((ps.vy0 + ps.vy)
                    + (ps.ay0 - ps.ay) * tau/6) * tau/2)
        ps.rz[:] = (ps.rz0
                    + ((ps.vz0 + ps.vz)
                    + (ps.az0 - ps.az) * tau/6) * tau/2)

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
        if "ahermite" in self.method:
            tau = self.get_hermite_tstep(ps, self.eta, tau)
        ps = self.pec(2, ps, tau)

        type(ps).t_curr += tau
        ps.tstep[:] = tau
        ps.time += tau
        ps.nstep += 1
        slc = ps.time % (self.dump_freq * tau) == 0
        if any(slc):
            self.wl.append(ps[slc].copy())
        if self.viewer:
            slc = ps.time % (self.gl_freq * tau) == 0
            if any(slc):
                self.viewer.show_event(ps[slc])
        return ps


########## end of file ##########
