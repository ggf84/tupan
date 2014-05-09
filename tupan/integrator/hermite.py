# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import logging
from ..integrator import Base
from ..lib.utils.timing import timings, bind_all


__all__ = ["Hermite"]

LOGGER = logging.getLogger(__name__)


@bind_all(timings)
class H2(object):
    """

    """
    @staticmethod
    def epredict(ps, dt):
        """

        """
        ps0 = ps.copy()
        ps0.set_acc(ps0)
        ps1 = ps

        ps1.rx += (ps0.ax * dt / 2 + ps0.vx) * dt
        ps1.ry += (ps0.ay * dt / 2 + ps0.vy) * dt
        ps1.rz += (ps0.az * dt / 2 + ps0.vz) * dt
        ps1.vx += ps0.ax * dt
        ps1.vy += ps0.ay * dt
        ps1.vz += ps0.az * dt

        return ps1, ps0

    @staticmethod
    def ecorrect(ps1, ps0, dt):
        """

        """
        ps1.set_acc(ps1)

        ps1.vx[...] = ((ps0.ax + ps1.ax) * dt / 2 + ps0.vx)
        ps1.vy[...] = ((ps0.ay + ps1.ay) * dt / 2 + ps0.vy)
        ps1.vz[...] = ((ps0.az + ps1.az) * dt / 2 + ps0.vz)

        ps1.rx[...] = ((ps0.vx + ps1.vx) * dt / 2 + ps0.rx)
        ps1.ry[...] = ((ps0.vy + ps1.vy) * dt / 2 + ps0.ry)
        ps1.rz[...] = ((ps0.vz + ps1.vz) * dt / 2 + ps0.rz)

        return ps1

    @classmethod
    def epec(cls, n, ps, dt):
        """

        """
        (ps1, ps0) = cls.epredict(ps, dt)
        for _ in range(n):
            ps1 = cls.ecorrect(ps1, ps0, dt)
        return ps1


@bind_all(timings)
class H4(H2):
    """

    """
    @staticmethod
    def epredict(ps, dt):
        """

        """
        ps0 = ps.copy()
        ps0.set_acc_jerk(ps0)
        ps1 = ps

        ps1.rx += ((ps0.jx * dt / 3 + ps0.ax) * dt / 2 + ps0.vx) * dt
        ps1.ry += ((ps0.jy * dt / 3 + ps0.ay) * dt / 2 + ps0.vy) * dt
        ps1.rz += ((ps0.jz * dt / 3 + ps0.az) * dt / 2 + ps0.vz) * dt
        ps1.vx += (ps0.jx * dt / 2 + ps0.ax) * dt
        ps1.vy += (ps0.jy * dt / 2 + ps0.ay) * dt
        ps1.vz += (ps0.jz * dt / 2 + ps0.az) * dt

        return ps1, ps0

    @staticmethod
    def ecorrect(ps1, ps0, dt):
        """

        """
        ps1.set_acc_jerk(ps1)

        ps1.vx[...] = (((ps0.jx - ps1.jx) * dt / 6
                        + (ps0.ax + ps1.ax)) * dt / 2
                       + ps0.vx)
        ps1.vy[...] = (((ps0.jy - ps1.jy) * dt / 6
                        + (ps0.ay + ps1.ay)) * dt / 2
                       + ps0.vy)
        ps1.vz[...] = (((ps0.jz - ps1.jz) * dt / 6
                        + (ps0.az + ps1.az)) * dt / 2
                       + ps0.vz)

        ps1.rx[...] = (((ps0.ax - ps1.ax) * dt / 6
                        + (ps0.vx + ps1.vx)) * dt / 2
                       + ps0.rx)
        ps1.ry[...] = (((ps0.ay - ps1.ay) * dt / 6
                        + (ps0.vy + ps1.vy)) * dt / 2
                       + ps0.ry)
        ps1.rz[...] = (((ps0.az - ps1.az) * dt / 6
                        + (ps0.vz + ps1.vz)) * dt / 2
                       + ps0.rz)

        return ps1


@bind_all(timings)
class H6(H4):
    """

    """
    @staticmethod
    def epredict(ps, dt):
        """

        """
        ps0 = ps.copy()
        ps0.set_acc_jerk(ps0)
        ps0.set_snap_crackle(ps0)
        ps1 = ps

        ps1.rx += (((ps0.sx * dt / 4
                     + ps0.jx) * dt / 3
                    + ps0.ax) * dt / 2
                   + ps0.vx) * dt
        ps1.ry += (((ps0.sy * dt / 4
                     + ps0.jy) * dt / 3
                    + ps0.ay) * dt / 2
                   + ps0.vy) * dt
        ps1.rz += (((ps0.sz * dt / 4
                     + ps0.jz) * dt / 3
                    + ps0.az) * dt / 2
                   + ps0.vz) * dt

        ps1.vx += ((ps0.sx * dt / 3
                    + ps0.jx) * dt / 2
                   + ps0.ax) * dt
        ps1.vy += ((ps0.sy * dt / 3
                    + ps0.jy) * dt / 2
                   + ps0.ay) * dt
        ps1.vz += ((ps0.sz * dt / 3
                    + ps0.jz) * dt / 2
                   + ps0.az) * dt

        return ps1, ps0

    @staticmethod
    def ecorrect(ps1, ps0, dt):
        """

        """
        ps1.set_acc_jerk(ps1)
        ps1.set_snap_crackle(ps1)

        ps1.vx[...] = ((((ps0.sx + ps1.sx) * dt / 12
                         + (ps0.jx - ps1.jx)) * dt / 5
                        + (ps0.ax + ps1.ax)) * dt / 2
                       + ps0.vx)
        ps1.vy[...] = ((((ps0.sy + ps1.sy) * dt / 12
                         + (ps0.jy - ps1.jy)) * dt / 5
                        + (ps0.ay + ps1.ay)) * dt / 2
                       + ps0.vy)
        ps1.vz[...] = ((((ps0.sz + ps1.sz) * dt / 12
                         + (ps0.jz - ps1.jz)) * dt / 5
                        + (ps0.az + ps1.az)) * dt / 2
                       + ps0.vz)

        ps1.rx[...] = ((((ps0.jx + ps1.jx) * dt / 12
                         + (ps0.ax - ps1.ax)) * dt / 5
                        + (ps0.vx + ps1.vx)) * dt / 2
                       + ps0.rx)
        ps1.ry[...] = ((((ps0.jy + ps1.jy) * dt / 12
                         + (ps0.ay - ps1.ay)) * dt / 5
                        + (ps0.vy + ps1.vy)) * dt / 2
                       + ps0.ry)
        ps1.rz[...] = ((((ps0.jz + ps1.jz) * dt / 12
                         + (ps0.az - ps1.az)) * dt / 5
                        + (ps0.vz + ps1.vz)) * dt / 2
                       + ps0.rz)

        return ps1


@bind_all(timings)
class H8(H6):
    """

    """
    @staticmethod
    def epredict(ps, dt):
        """

        """
        ps0 = ps.copy()
        ps0.set_acc_jerk(ps0)
        ps0.set_snap_crackle(ps0)
        ps1 = ps

        ps1.rx += ((((ps0.cx * dt / 5
                      + ps0.sx) * dt / 4
                     + ps0.jx) * dt / 3
                    + ps0.ax) * dt / 2
                   + ps0.vx) * dt
        ps1.ry += ((((ps0.cy * dt / 5
                      + ps0.sy) * dt / 4
                     + ps0.jy) * dt / 3
                    + ps0.ay) * dt / 2
                   + ps0.vy) * dt
        ps1.rz += ((((ps0.cz * dt / 5
                      + ps0.sz) * dt / 4
                     + ps0.jz) * dt / 3
                    + ps0.az) * dt / 2
                   + ps0.vz) * dt

        ps1.vx += (((ps0.cx * dt / 4
                     + ps0.sx) * dt / 3
                    + ps0.jx) * dt / 2
                   + ps0.ax) * dt
        ps1.vy += (((ps0.cy * dt / 4
                     + ps0.sy) * dt / 3
                    + ps0.jy) * dt / 2
                   + ps0.ay) * dt
        ps1.vz += (((ps0.cz * dt / 4
                     + ps0.sz) * dt / 3
                    + ps0.jz) * dt / 2
                   + ps0.az) * dt

        return ps1, ps0

    @staticmethod
    def ecorrect(ps1, ps0, dt):
        """

        """
        ps1.set_acc_jerk(ps1)
        ps1.set_snap_crackle(ps1)

        ps1.vx[...] = (((((ps0.cx - ps1.cx) * dt / 20
                          + (ps0.sx + ps1.sx)) * dt / 3
                         + 3 * (ps0.jx - ps1.jx)) * dt / 14
                        + (ps0.ax + ps1.ax)) * dt / 2
                       + ps0.vx)
        ps1.vy[...] = (((((ps0.cy - ps1.cy) * dt / 20
                          + (ps0.sy + ps1.sy)) * dt / 3
                         + 3 * (ps0.jy - ps1.jy)) * dt / 14
                        + (ps0.ay + ps1.ay)) * dt / 2
                       + ps0.vy)
        ps1.vz[...] = (((((ps0.cz - ps1.cz) * dt / 20
                          + (ps0.sz + ps1.sz)) * dt / 3
                         + 3 * (ps0.jz - ps1.jz)) * dt / 14
                        + (ps0.az + ps1.az)) * dt / 2
                       + ps0.vz)

        ps1.rx[...] = (((((ps0.sx - ps1.sx) * dt / 20
                          + (ps0.jx + ps1.jx)) * dt / 3
                         + 3 * (ps0.ax - ps1.ax)) * dt / 14
                        + (ps0.vx + ps1.vx)) * dt / 2
                       + ps0.rx)
        ps1.ry[...] = (((((ps0.sy - ps1.sy) * dt / 20
                          + (ps0.jy + ps1.jy)) * dt / 3
                         + 3 * (ps0.ay - ps1.ay)) * dt / 14
                        + (ps0.vy + ps1.vy)) * dt / 2
                       + ps0.ry)
        ps1.rz[...] = (((((ps0.sz - ps1.sz) * dt / 20
                          + (ps0.jz + ps1.jz)) * dt / 3
                         + 3 * (ps0.az - ps1.az)) * dt / 14
                        + (ps0.vz + ps1.vz)) * dt / 2
                       + ps0.rz)

        return ps1


@bind_all(timings)
class Hermite(Base):
    """

    """
    PROVIDED_METHODS = ['hermite2', 'ahermite2',
                        'hermite4', 'ahermite4',
                        'hermite6', 'ahermite6',
                        'hermite8', 'ahermite8', ]

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
        ps = self.ps
        LOGGER.info("Initializing '%s' integrator at "
                    "t_curr = %g and t_end = %g.",
                    self.method, ps.t_curr, t_end)

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
        ps = self.ps
        LOGGER.info("Finalizing '%s' integrator at "
                    "t_curr = %g and t_end = %g.",
                    self.method, ps.t_curr, t_end)

        if self.viewer:
            self.viewer.show_event(ps)
            self.viewer.enter_main_loop()

    def get_hermite_tstep(self, ps, eta, dt):
        """

        """
        ps.set_tstep(ps, eta)
        min_bts = self.get_min_block_tstep(ps, dt)
        return min_bts

    def epredict(self, ps, dt):
        """

        """
        if self.order == 2:
            return H2.epredict(ps, dt)
        elif self.order == 4:
            return H4.epredict(ps, dt)
        elif self.order == 6:
            return H6.epredict(ps, dt)
        elif self.order == 8:
            return H8.epredict(ps, dt)

    def ecorrect(self, ps1, ps0, dt):
        """

        """
        if self.order == 2:
            return H2.ecorrect(ps1, ps0, dt)
        elif self.order == 4:
            return H4.ecorrect(ps1, ps0, dt)
        elif self.order == 6:
            return H6.ecorrect(ps1, ps0, dt)
        elif self.order == 8:
            return H8.ecorrect(ps1, ps0, dt)

    def epec(self, n, ps, dt):
        """

        """
        if self.order == 2:
            return H2.epec(n, ps, dt)
        elif self.order == 4:
            return H4.epec(n, ps, dt)
        elif self.order == 6:
            return H6.epec(n, ps, dt)
        elif self.order == 8:
            return H8.epec(n, ps, dt)

    def do_step(self, ps, dt):
        """

        """
        if "ahermite" in self.method:
            dt = self.get_hermite_tstep(ps, self.eta, dt)
        ps = self.epec(2, ps, dt)

        type(ps).t_curr += dt
        ps.tstep[...] = dt
        ps.time += dt
        ps.nstep += 1
        if self.dumpper:
            slc = ps.time % (self.dump_freq * dt) == 0
            if any(slc):
                self.wl.append(ps[slc])
        if self.viewer:
            slc = ps.time % (self.gl_freq * dt) == 0
            if any(slc):
                self.viewer.show_event(ps[slc])
        return ps


# -- End of File --
