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
    def epredict(ps, tau):
        """

        """
        ps0 = ps.copy()
        ps0.set_acc(ps0)
        ps1 = ps

        ps1.rx += (ps0.ax * tau / 2 + ps0.vx) * tau
        ps1.ry += (ps0.ay * tau / 2 + ps0.vy) * tau
        ps1.rz += (ps0.az * tau / 2 + ps0.vz) * tau
        ps1.vx += ps0.ax * tau
        ps1.vy += ps0.ay * tau
        ps1.vz += ps0.az * tau

        return ps1, ps0

    @staticmethod
    def ecorrect(ps1, ps0, tau):
        """

        """
        ps1.set_acc(ps1)

        ps1.vx[...] = ((ps0.ax + ps1.ax) * tau / 2 + ps0.vx)
        ps1.vy[...] = ((ps0.ay + ps1.ay) * tau / 2 + ps0.vy)
        ps1.vz[...] = ((ps0.az + ps1.az) * tau / 2 + ps0.vz)

        ps1.rx[...] = ((ps0.vx + ps1.vx) * tau / 2 + ps0.rx)
        ps1.ry[...] = ((ps0.vy + ps1.vy) * tau / 2 + ps0.ry)
        ps1.rz[...] = ((ps0.vz + ps1.vz) * tau / 2 + ps0.rz)

        return ps1

    @classmethod
    def epec(cls, n, ps, tau):
        """

        """
        (ps1, ps0) = cls.epredict(ps, tau)
        for i in range(n):
            ps1 = cls.ecorrect(ps1, ps0, tau)
        return ps1


class H4(H2):
    """

    """
    @staticmethod
    def epredict(ps, tau):
        """

        """
        ps0 = ps.copy()
        ps0.set_acc_jerk(ps0)
        ps1 = ps

        ps1.rx += ((ps0.jx * tau / 3 + ps0.ax) * tau / 2 + ps0.vx) * tau
        ps1.ry += ((ps0.jy * tau / 3 + ps0.ay) * tau / 2 + ps0.vy) * tau
        ps1.rz += ((ps0.jz * tau / 3 + ps0.az) * tau / 2 + ps0.vz) * tau
        ps1.vx += (ps0.jx * tau / 2 + ps0.ax) * tau
        ps1.vy += (ps0.jy * tau / 2 + ps0.ay) * tau
        ps1.vz += (ps0.jz * tau / 2 + ps0.az) * tau

        return ps1, ps0

    @staticmethod
    def ecorrect(ps1, ps0, tau):
        """

        """
        ps1.set_acc_jerk(ps1)

        ps1.vx[...] = (((ps0.jx - ps1.jx) * tau / 6
                       + (ps0.ax + ps1.ax)) * tau / 2
                       + ps0.vx)
        ps1.vy[...] = (((ps0.jy - ps1.jy) * tau / 6
                       + (ps0.ay + ps1.ay)) * tau / 2
                       + ps0.vy)
        ps1.vz[...] = (((ps0.jz - ps1.jz) * tau / 6
                       + (ps0.az + ps1.az)) * tau / 2
                       + ps0.vz)

        ps1.rx[...] = (((ps0.ax - ps1.ax) * tau / 6
                       + (ps0.vx + ps1.vx)) * tau / 2
                       + ps0.rx)
        ps1.ry[...] = (((ps0.ay - ps1.ay) * tau / 6
                       + (ps0.vy + ps1.vy)) * tau / 2
                       + ps0.ry)
        ps1.rz[...] = (((ps0.az - ps1.az) * tau / 6
                       + (ps0.vz + ps1.vz)) * tau / 2
                       + ps0.rz)

        return ps1


class H6(H4):
    """

    """
    @staticmethod
    def epredict(ps, tau):
        """

        """
        ps0 = ps.copy()
        ps0.set_acc_jerk(ps0)
        ps0.set_snap_crackle(ps0)
        ps1 = ps

        ps1.rx += (((ps0.sx * tau / 4
                   + ps0.jx) * tau / 3
                   + ps0.ax) * tau / 2
                   + ps0.vx) * tau
        ps1.ry += (((ps0.sy * tau / 4
                   + ps0.jy) * tau / 3
                   + ps0.ay) * tau / 2
                   + ps0.vy) * tau
        ps1.rz += (((ps0.sz * tau / 4
                   + ps0.jz) * tau / 3
                   + ps0.az) * tau / 2
                   + ps0.vz) * tau

        ps1.vx += ((ps0.sx * tau / 3
                   + ps0.jx) * tau / 2
                   + ps0.ax) * tau
        ps1.vy += ((ps0.sy * tau / 3
                   + ps0.jy) * tau / 2
                   + ps0.ay) * tau
        ps1.vz += ((ps0.sz * tau / 3
                   + ps0.jz) * tau / 2
                   + ps0.az) * tau

        return ps1, ps0

    @staticmethod
    def ecorrect(ps1, ps0, tau):
        """

        """
        ps1.set_acc_jerk(ps1)
        ps1.set_snap_crackle(ps1)

        ps1.vx[...] = ((((ps0.sx + ps1.sx) * tau / 12
                       + (ps0.jx - ps1.jx)) * tau / 5
                       + (ps0.ax + ps1.ax)) * tau / 2
                       + ps0.vx)
        ps1.vy[...] = ((((ps0.sy + ps1.sy) * tau / 12
                       + (ps0.jy - ps1.jy)) * tau / 5
                       + (ps0.ay + ps1.ay)) * tau / 2
                       + ps0.vy)
        ps1.vz[...] = ((((ps0.sz + ps1.sz) * tau / 12
                       + (ps0.jz - ps1.jz)) * tau / 5
                       + (ps0.az + ps1.az)) * tau / 2
                       + ps0.vz)

        ps1.rx[...] = ((((ps0.jx + ps1.jx) * tau / 12
                       + (ps0.ax - ps1.ax)) * tau / 5
                       + (ps0.vx + ps1.vx)) * tau / 2
                       + ps0.rx)
        ps1.ry[...] = ((((ps0.jy + ps1.jy) * tau / 12
                       + (ps0.ay - ps1.ay)) * tau / 5
                       + (ps0.vy + ps1.vy)) * tau / 2
                       + ps0.ry)
        ps1.rz[...] = ((((ps0.jz + ps1.jz) * tau / 12
                       + (ps0.az - ps1.az)) * tau / 5
                       + (ps0.vz + ps1.vz)) * tau / 2
                       + ps0.rz)

        return ps1


class H8(H6):
    """

    """
    @staticmethod
    def epredict(ps, tau):
        """

        """
        ps0 = ps.copy()
        ps0.set_acc_jerk(ps0)
        ps0.set_snap_crackle(ps0)
        ps1 = ps

        ps1.rx += ((((ps0.cx * tau / 5
                   + ps0.sx) * tau / 4
                   + ps0.jx) * tau / 3
                   + ps0.ax) * tau / 2
                   + ps0.vx) * tau
        ps1.ry += ((((ps0.cy * tau / 5
                   + ps0.sy) * tau / 4
                   + ps0.jy) * tau / 3
                   + ps0.ay) * tau / 2
                   + ps0.vy) * tau
        ps1.rz += ((((ps0.cz * tau / 5
                   + ps0.sz) * tau / 4
                   + ps0.jz) * tau / 3
                   + ps0.az) * tau / 2
                   + ps0.vz) * tau

        ps1.vx += (((ps0.cx * tau / 4
                   + ps0.sx) * tau / 3
                   + ps0.jx) * tau / 2
                   + ps0.ax) * tau
        ps1.vy += (((ps0.cy * tau / 4
                   + ps0.sy) * tau / 3
                   + ps0.jy) * tau / 2
                   + ps0.ay) * tau
        ps1.vz += (((ps0.cz * tau / 4
                   + ps0.sz) * tau / 3
                   + ps0.jz) * tau / 2
                   + ps0.az) * tau

        return ps1, ps0

    @staticmethod
    def ecorrect(ps1, ps0, tau):
        """

        """
        ps1.set_acc_jerk(ps1)
        ps1.set_snap_crackle(ps1)

        ps1.vx[...] = (((((ps0.cx - ps1.cx) * tau / 20
                       + (ps0.sx + ps1.sx)) * tau / 3
                       + 3 * (ps0.jx - ps1.jx)) * tau / 14
                       + (ps0.ax + ps1.ax)) * tau / 2
                       + ps0.vx)
        ps1.vy[...] = (((((ps0.cy - ps1.cy) * tau / 20
                       + (ps0.sy + ps1.sy)) * tau / 3
                       + 3 * (ps0.jy - ps1.jy)) * tau / 14
                       + (ps0.ay + ps1.ay)) * tau / 2
                       + ps0.vy)
        ps1.vz[...] = (((((ps0.cz - ps1.cz) * tau / 20
                       + (ps0.sz + ps1.sz)) * tau / 3
                       + 3 * (ps0.jz - ps1.jz)) * tau / 14
                       + (ps0.az + ps1.az)) * tau / 2
                       + ps0.vz)

        ps1.rx[...] = (((((ps0.sx - ps1.sx) * tau / 20
                       + (ps0.jx + ps1.jx)) * tau / 3
                       + 3 * (ps0.ax - ps1.ax)) * tau / 14
                       + (ps0.vx + ps1.vx)) * tau / 2
                       + ps0.rx)
        ps1.ry[...] = (((((ps0.sy - ps1.sy) * tau / 20
                       + (ps0.jy + ps1.jy)) * tau / 3
                       + 3 * (ps0.ay - ps1.ay)) * tau / 14
                       + (ps0.vy + ps1.vy)) * tau / 2
                       + ps0.ry)
        ps1.rz[...] = (((((ps0.sz - ps1.sz) * tau / 20
                       + (ps0.jz + ps1.jz)) * tau / 3
                       + 3 * (ps0.az - ps1.az)) * tau / 14
                       + (ps0.vz + ps1.vz)) * tau / 2
                       + ps0.rz)

        return ps1


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

    def epredict(self, ps, tau):
        """

        """
        if self.order == 2:
            return H2.epredict(ps, tau)
        elif self.order == 4:
            return H4.epredict(ps, tau)
        elif self.order == 6:
            return H6.epredict(ps, tau)
        elif self.order == 8:
            return H8.epredict(ps, tau)

    def ecorrect(self, ps1, ps0, tau):
        """

        """
        if self.order == 2:
            return H2.ecorrect(ps1, ps0, tau)
        elif self.order == 4:
            return H4.ecorrect(ps1, ps0, tau)
        elif self.order == 6:
            return H6.ecorrect(ps1, ps0, tau)
        elif self.order == 8:
            return H8.ecorrect(ps1, ps0, tau)

    def epec(self, n, ps, tau):
        """

        """
        if self.order == 2:
            return H2.epec(n, ps, tau)
        elif self.order == 4:
            return H4.epec(n, ps, tau)
        elif self.order == 6:
            return H6.epec(n, ps, tau)
        elif self.order == 8:
            return H8.epec(n, ps, tau)

    def do_step(self, ps, tau):
        """

        """
        if "ahermite" in self.method:
            tau = self.get_hermite_tstep(ps, self.eta, tau)
        ps = self.epec(2, ps, tau)

        type(ps).t_curr += tau
        ps.tstep[...] = tau
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
