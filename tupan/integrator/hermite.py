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
    PROVIDED_METHODS = ['hermite2', 'ahermite2',
                        'hermite4', 'ahermite4',
                        'hermite6', 'ahermite6',
                        'hermite8', 'ahermite8',
                        ]

    def __init__(self, eta, time, ps, method, **kwargs):
        """

        """
        super(Hermite, self).__init__(eta, time, ps, **kwargs)
        self.method = method
        if 'hermite2' in self.method:
            self.order = 2
        if 'hermite4' in self.method:
            self.order = 4
        if 'hermite6' in self.method:
            self.order = 6
        if 'hermite8' in self.method:
            self.order = 8

    def initialize(self, t_end):
        """

        """
        logger.info("Initializing '%s' integrator.",
                    self.method)

        ps = self.ps
        if self.order == 2:
            ps.set_acc(ps)
        if self.order >= 4:
            ps.set_acc_jerk(ps)
        if self.order >= 6:
            ps.set_snap_crackle(ps)

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
        ps.set_tstep(ps, eta)
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
        if self.order >= 2:
            ps.ax0 = ps.ax.copy()
            ps.ay0 = ps.ay.copy()
            ps.az0 = ps.az.copy()
        if self.order >= 4:
            ps.jx0 = ps.jx.copy()
            ps.jy0 = ps.jy.copy()
            ps.jz0 = ps.jz.copy()
        if self.order >= 6:
            ps.sx0 = ps.sx.copy()
            ps.sy0 = ps.sy.copy()
            ps.sz0 = ps.sz.copy()
            ps.cx0 = ps.cx.copy()
            ps.cy0 = ps.cy.copy()
            ps.cz0 = ps.cz.copy()

        drx = 0.0
        dry = 0.0
        drz = 0.0
        dvx = 0.0
        dvy = 0.0
        dvz = 0.0
        if self.order >= 6:
            drx = ((drx + ps.cx) * tau / 5 + ps.sx) * tau / 4
            dry = ((dry + ps.cy) * tau / 5 + ps.sy) * tau / 4
            drz = ((drz + ps.cz) * tau / 5 + ps.sz) * tau / 4
            dvx = ((dvx + ps.cx) * tau / 4 + ps.sx) * tau / 3
            dvy = ((dvy + ps.cy) * tau / 4 + ps.sy) * tau / 3
            dvz = ((dvz + ps.cz) * tau / 4 + ps.sz) * tau / 3

        if self.order >= 4:
            drx = ((drx + ps.jx) * tau / 3 + ps.ax) * tau / 2
            dry = ((dry + ps.jy) * tau / 3 + ps.ay) * tau / 2
            drz = ((drz + ps.jz) * tau / 3 + ps.az) * tau / 2
            dvx = ((dvx + ps.jx) * tau / 2 + ps.ax) * tau
            dvy = ((dvy + ps.jy) * tau / 2 + ps.ay) * tau
            dvz = ((dvz + ps.jz) * tau / 2 + ps.az) * tau

        drx = (drx + ps.vx) * tau
        dry = (dry + ps.vy) * tau
        drz = (drz + ps.vz) * tau

        ps.rx += drx
        ps.ry += dry
        ps.rz += drz
        ps.vx += dvx
        ps.vy += dvy
        ps.vz += dvz

    def ecorrect(self, ps, tau):
        """

        """
        if self.order == 2:
            ps.set_acc(ps)
        if self.order >= 4:
            ps.set_acc_jerk(ps)
        if self.order >= 6:
            ps.set_snap_crackle(ps)

        drx = 0.0
        dry = 0.0
        drz = 0.0
        dvx = 0.0
        dvy = 0.0
        dvz = 0.0
        if self.order == 8:
            drx = (((drx + (ps.sx0 - ps.sx)) * tau / 20
                   + (ps.jx0 + ps.jx)) * tau / 3
                   + 3 * (ps.ax0 - ps.ax)) * tau / 14
            dry = (((dry + (ps.sy0 - ps.sy)) * tau / 20
                   + (ps.jy0 + ps.jy)) * tau / 3
                   + 3 * (ps.ay0 - ps.ay)) * tau / 14
            drz = (((drz + (ps.sz0 - ps.sz)) * tau / 20
                   + (ps.jz0 + ps.jz)) * tau / 3
                   + 3 * (ps.az0 - ps.az)) * tau / 14
            dvx = (((dvx + (ps.cx0 - ps.cx)) * tau / 20
                   + (ps.sx0 + ps.sx)) * tau / 3
                   + 3 * (ps.jx0 - ps.jx)) * tau / 14
            dvy = (((dvy + (ps.cy0 - ps.cy)) * tau / 20
                   + (ps.sy0 + ps.sy)) * tau / 3
                   + 3 * (ps.jy0 - ps.jy)) * tau / 14
            dvz = (((dvz + (ps.cz0 - ps.cz)) * tau / 20
                   + (ps.sz0 + ps.sz)) * tau / 3
                   + 3 * (ps.jz0 - ps.jz)) * tau / 14

        if self.order == 6:
            drx = ((drx + (ps.jx0 + ps.jx)) * tau / 12
                   + (ps.ax0 - ps.ax)) * tau / 5
            dry = ((dry + (ps.jy0 + ps.jy)) * tau / 12
                   + (ps.ay0 - ps.ay)) * tau / 5
            drz = ((drz + (ps.jz0 + ps.jz)) * tau / 12
                   + (ps.az0 - ps.az)) * tau / 5
            dvx = ((dvx + (ps.sx0 + ps.sx)) * tau / 12
                   + (ps.jx0 - ps.jx)) * tau / 5
            dvy = ((dvy + (ps.sy0 + ps.sy)) * tau / 12
                   + (ps.jy0 - ps.jy)) * tau / 5
            dvz = ((dvz + (ps.sz0 + ps.sz)) * tau / 12
                   + (ps.jz0 - ps.jz)) * tau / 5

        if self.order == 4:
            drx = (drx + (ps.ax0 - ps.ax)) * tau / 6
            dry = (dry + (ps.ay0 - ps.ay)) * tau / 6
            drz = (drz + (ps.az0 - ps.az)) * tau / 6
            dvx = (dvx + (ps.jx0 - ps.jx)) * tau / 6
            dvy = (dvy + (ps.jy0 - ps.jy)) * tau / 6
            dvz = (dvz + (ps.jz0 - ps.jz)) * tau / 6

        dvx = (dvx + (ps.ax0 + ps.ax)) * tau / 2
        dvy = (dvy + (ps.ay0 + ps.ay)) * tau / 2
        dvz = (dvz + (ps.az0 + ps.az)) * tau / 2

        ps.vx[:] = ps.vx0 + dvx
        ps.vy[:] = ps.vy0 + dvy
        ps.vz[:] = ps.vz0 + dvz

        drx = (drx + (ps.vx0 + ps.vx)) * tau / 2
        dry = (dry + (ps.vy0 + ps.vy)) * tau / 2
        drz = (drz + (ps.vz0 + ps.vz)) * tau / 2

        ps.rx[:] = ps.rx0 + drx
        ps.ry[:] = ps.ry0 + dry
        ps.rz[:] = ps.rz0 + drz

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
        if self.dumpper:
            slc = ps.time % (self.dump_freq * tau) == 0
            if any(slc):
                self.wl.append(ps[slc])
        if self.viewer:
            slc = ps.time % (self.gl_freq * tau) == 0
            if any(slc):
                self.viewer.show_event(ps[slc])
        return ps


########## end of file ##########
