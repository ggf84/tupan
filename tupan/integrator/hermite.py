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


class H2(object):
    """

    """
    @staticmethod
    def predict(ps, tau):
        """

        """
        ps.rx0[:], ps.ry0[:], ps.rz0[:] = ps.rx, ps.ry, ps.rz
        ps.vx0[:], ps.vy0[:], ps.vz0[:] = ps.vx, ps.vy, ps.vz
        ps.ax0[:], ps.ay0[:], ps.az0[:] = ps.ax, ps.ay, ps.az

        ps.rx += (ps.ax * tau / 2 + ps.vx) * tau
        ps.ry += (ps.ay * tau / 2 + ps.vy) * tau
        ps.rz += (ps.az * tau / 2 + ps.vz) * tau
        ps.vx += ps.ax * tau
        ps.vy += ps.ay * tau
        ps.vz += ps.az * tau

        return ps

    @staticmethod
    def ecorrect(ps, tau):
        """

        """
        ps.set_acc(ps)

        ps.vx[:] = ((ps.ax0 + ps.ax) * tau / 2 + ps.vx0)
        ps.vy[:] = ((ps.ay0 + ps.ay) * tau / 2 + ps.vy0)
        ps.vz[:] = ((ps.az0 + ps.az) * tau / 2 + ps.vz0)

        ps.rx[:] = ((ps.vx0 + ps.vx) * tau / 2 + ps.rx0)
        ps.ry[:] = ((ps.vy0 + ps.vy) * tau / 2 + ps.ry0)
        ps.rz[:] = ((ps.vz0 + ps.vz) * tau / 2 + ps.rz0)

        return ps


class H4(object):
    """

    """
    @staticmethod
    def predict(ps, tau):
        """

        """
        ps.rx0[:], ps.ry0[:], ps.rz0[:] = ps.rx, ps.ry, ps.rz
        ps.vx0[:], ps.vy0[:], ps.vz0[:] = ps.vx, ps.vy, ps.vz
        ps.ax0[:], ps.ay0[:], ps.az0[:] = ps.ax, ps.ay, ps.az
        ps.jx0[:], ps.jy0[:], ps.jz0[:] = ps.jx, ps.jy, ps.jz

        ps.rx += ((ps.jx * tau / 3 + ps.ax) * tau / 2 + ps.vx) * tau
        ps.ry += ((ps.jy * tau / 3 + ps.ay) * tau / 2 + ps.vy) * tau
        ps.rz += ((ps.jz * tau / 3 + ps.az) * tau / 2 + ps.vz) * tau
        ps.vx += (ps.jx * tau / 2 + ps.ax) * tau
        ps.vy += (ps.jy * tau / 2 + ps.ay) * tau
        ps.vz += (ps.jz * tau / 2 + ps.az) * tau

        return ps

    @staticmethod
    def ecorrect(ps, tau):
        """

        """
        ps.set_acc_jerk(ps)

        ps.vx[:] = (((ps.jx0 - ps.jx) * tau / 6
                    + (ps.ax0 + ps.ax)) * tau / 2
                    + ps.vx0)
        ps.vy[:] = (((ps.jy0 - ps.jy) * tau / 6
                    + (ps.ay0 + ps.ay)) * tau / 2
                    + ps.vy0)
        ps.vz[:] = (((ps.jz0 - ps.jz) * tau / 6
                    + (ps.az0 + ps.az)) * tau / 2
                    + ps.vz0)

        ps.rx[:] = (((ps.ax0 - ps.ax) * tau / 6
                    + (ps.vx0 + ps.vx)) * tau / 2
                    + ps.rx0)
        ps.ry[:] = (((ps.ay0 - ps.ay) * tau / 6
                    + (ps.vy0 + ps.vy)) * tau / 2
                    + ps.ry0)
        ps.rz[:] = (((ps.az0 - ps.az) * tau / 6
                    + (ps.vz0 + ps.vz)) * tau / 2
                    + ps.rz0)

        return ps


class H6(object):
    """

    """
    @staticmethod
    def predict(ps, tau):
        """

        """
        ps.rx0[:], ps.ry0[:], ps.rz0[:] = ps.rx, ps.ry, ps.rz
        ps.vx0[:], ps.vy0[:], ps.vz0[:] = ps.vx, ps.vy, ps.vz
        ps.ax0[:], ps.ay0[:], ps.az0[:] = ps.ax, ps.ay, ps.az
        ps.jx0[:], ps.jy0[:], ps.jz0[:] = ps.jx, ps.jy, ps.jz
        ps.sx0[:], ps.sy0[:], ps.sz0[:] = ps.sx, ps.sy, ps.sz

        ps.rx += (((ps.sx * tau / 4
                  + ps.jx) * tau / 3
                  + ps.ax) * tau / 2
                  + ps.vx) * tau
        ps.ry += (((ps.sy * tau / 4
                  + ps.jy) * tau / 3
                  + ps.ay) * tau / 2
                  + ps.vy) * tau
        ps.rz += (((ps.sz * tau / 4
                  + ps.jz) * tau / 3
                  + ps.az) * tau / 2
                  + ps.vz) * tau

        ps.vx += ((ps.sx * tau / 3
                  + ps.jx) * tau / 2
                  + ps.ax) * tau
        ps.vy += ((ps.sy * tau / 3
                  + ps.jy) * tau / 2
                  + ps.ay) * tau
        ps.vz += ((ps.sz * tau / 3
                  + ps.jz) * tau / 2
                  + ps.az) * tau

        return ps

    @staticmethod
    def ecorrect(ps, tau):
        """

        """
        ps.set_acc_jerk(ps)
        ps.set_snap_crackle(ps)

        ps.vx[:] = ((((ps.sx0 + ps.sx) * tau / 12
                    + (ps.jx0 - ps.jx)) * tau / 5
                    + (ps.ax0 + ps.ax)) * tau / 2
                    + ps.vx0)
        ps.vy[:] = ((((ps.sy0 + ps.sy) * tau / 12
                    + (ps.jy0 - ps.jy)) * tau / 5
                    + (ps.ay0 + ps.ay)) * tau / 2
                    + ps.vy0)
        ps.vz[:] = ((((ps.sz0 + ps.sz) * tau / 12
                    + (ps.jz0 - ps.jz)) * tau / 5
                    + (ps.az0 + ps.az)) * tau / 2
                    + ps.vz0)

        ps.rx[:] = ((((ps.jx0 + ps.jx) * tau / 12
                    + (ps.ax0 - ps.ax)) * tau / 5
                    + (ps.vx0 + ps.vx)) * tau / 2
                    + ps.rx0)
        ps.ry[:] = ((((ps.jy0 + ps.jy) * tau / 12
                    + (ps.ay0 - ps.ay)) * tau / 5
                    + (ps.vy0 + ps.vy)) * tau / 2
                    + ps.ry0)
        ps.rz[:] = ((((ps.jz0 + ps.jz) * tau / 12
                    + (ps.az0 - ps.az)) * tau / 5
                    + (ps.vz0 + ps.vz)) * tau / 2
                    + ps.rz0)

        return ps


class H8(object):
    """

    """
    @staticmethod
    def predict(ps, tau):
        """

        """
        ps.rx0[:], ps.ry0[:], ps.rz0[:] = ps.rx, ps.ry, ps.rz
        ps.vx0[:], ps.vy0[:], ps.vz0[:] = ps.vx, ps.vy, ps.vz
        ps.ax0[:], ps.ay0[:], ps.az0[:] = ps.ax, ps.ay, ps.az
        ps.jx0[:], ps.jy0[:], ps.jz0[:] = ps.jx, ps.jy, ps.jz
        ps.sx0[:], ps.sy0[:], ps.sz0[:] = ps.sx, ps.sy, ps.sz
        ps.cx0[:], ps.cy0[:], ps.cz0[:] = ps.cx, ps.cy, ps.cz

        ps.rx += ((((ps.cx * tau / 5
                  + ps.sx) * tau / 4
                  + ps.jx) * tau / 3
                  + ps.ax) * tau / 2
                  + ps.vx) * tau
        ps.ry += ((((ps.cy * tau / 5
                  + ps.sy) * tau / 4
                  + ps.jy) * tau / 3
                  + ps.ay) * tau / 2
                  + ps.vy) * tau
        ps.rz += ((((ps.cz * tau / 5
                  + ps.sz) * tau / 4
                  + ps.jz) * tau / 3
                  + ps.az) * tau / 2
                  + ps.vz) * tau

        ps.vx += (((ps.cx * tau / 4
                  + ps.sx) * tau / 3
                  + ps.jx) * tau / 2
                  + ps.ax) * tau
        ps.vy += (((ps.cy * tau / 4
                  + ps.sy) * tau / 3
                  + ps.jy) * tau / 2
                  + ps.ay) * tau
        ps.vz += (((ps.cz * tau / 4
                  + ps.sz) * tau / 3
                  + ps.jz) * tau / 2
                  + ps.az) * tau

        return ps

    @staticmethod
    def ecorrect(ps, tau):
        """

        """
        ps.set_acc_jerk(ps)
        ps.set_snap_crackle(ps)

        ps.vx[:] = (((((ps.cx0 - ps.cx) * tau / 20
                    + (ps.sx0 + ps.sx)) * tau / 3
                    + 3 * (ps.jx0 - ps.jx)) * tau / 14
                    + (ps.ax0 + ps.ax)) * tau / 2
                    + ps.vx0)
        ps.vy[:] = (((((ps.cy0 - ps.cy) * tau / 20
                    + (ps.sy0 + ps.sy)) * tau / 3
                    + 3 * (ps.jy0 - ps.jy)) * tau / 14
                    + (ps.ay0 + ps.ay)) * tau / 2
                    + ps.vy0)
        ps.vz[:] = (((((ps.cz0 - ps.cz) * tau / 20
                    + (ps.sz0 + ps.sz)) * tau / 3
                    + 3 * (ps.jz0 - ps.jz)) * tau / 14
                    + (ps.az0 + ps.az)) * tau / 2
                    + ps.vz0)

        ps.rx[:] = (((((ps.sx0 - ps.sx) * tau / 20
                    + (ps.jx0 + ps.jx)) * tau / 3
                    + 3 * (ps.ax0 - ps.ax)) * tau / 14
                    + (ps.vx0 + ps.vx)) * tau / 2
                    + ps.rx0)
        ps.ry[:] = (((((ps.sy0 - ps.sy) * tau / 20
                    + (ps.jy0 + ps.jy)) * tau / 3
                    + 3 * (ps.ay0 - ps.ay)) * tau / 14
                    + (ps.vy0 + ps.vy)) * tau / 2
                    + ps.ry0)
        ps.rz[:] = (((((ps.sz0 - ps.sz) * tau / 20
                    + (ps.jz0 + ps.jz)) * tau / 3
                    + 3 * (ps.az0 - ps.az)) * tau / 14
                    + (ps.vz0 + ps.vz)) * tau / 2
                    + ps.rz0)

        return ps


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
        ps.register_auxiliary_attribute("rx0", ps.rx.dtype)
        ps.register_auxiliary_attribute("ry0", ps.ry.dtype)
        ps.register_auxiliary_attribute("rz0", ps.rz.dtype)
        ps.register_auxiliary_attribute("vx0", ps.vx.dtype)
        ps.register_auxiliary_attribute("vy0", ps.vy.dtype)
        ps.register_auxiliary_attribute("vz0", ps.vz.dtype)
        if self.order >= 2:
            ps.set_acc(ps)
            ps.register_auxiliary_attribute("ax0", ps.ax.dtype)
            ps.register_auxiliary_attribute("ay0", ps.ay.dtype)
            ps.register_auxiliary_attribute("az0", ps.az.dtype)
        if self.order >= 4:
            ps.set_acc_jerk(ps)
            ps.register_auxiliary_attribute("jx0", ps.jx.dtype)
            ps.register_auxiliary_attribute("jy0", ps.jy.dtype)
            ps.register_auxiliary_attribute("jz0", ps.jz.dtype)
        if self.order >= 6:
            ps.set_snap_crackle(ps)
            ps.register_auxiliary_attribute("sx0", ps.sx.dtype)
            ps.register_auxiliary_attribute("sy0", ps.sy.dtype)
            ps.register_auxiliary_attribute("sz0", ps.sz.dtype)
        if self.order >= 8:
            ps.register_auxiliary_attribute("cx0", ps.cx.dtype)
            ps.register_auxiliary_attribute("cy0", ps.cy.dtype)
            ps.register_auxiliary_attribute("cz0", ps.cz.dtype)

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
        if self.order == 2:
            return H2.predict(ps, tau)
        elif self.order == 4:
            return H4.predict(ps, tau)
        elif self.order == 6:
            return H6.predict(ps, tau)
        elif self.order == 8:
            return H8.predict(ps, tau)

    def ecorrect(self, ps, tau):
        """

        """
        if self.order == 2:
            return H2.ecorrect(ps, tau)
        elif self.order == 4:
            return H4.ecorrect(ps, tau)
        elif self.order == 6:
            return H6.ecorrect(ps, tau)
        elif self.order == 8:
            return H8.ecorrect(ps, tau)

    def pec(self, n, ps, tau):
        """

        """
        ps = self.predict(ps, tau)
        for i in range(n):
            ps = self.ecorrect(ps, tau)
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
