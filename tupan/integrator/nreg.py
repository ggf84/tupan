# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function, division
import logging
from ..integrator import Base
from ..lib import extensions
from ..lib.utils.timing import decallmethods, timings


__all__ = ["NREG"]


logger = logging.getLogger(__name__)


def nreg_x(ps, dt):
    """

    """
    mtot = ps.total_mass
    extensions.nreg_x.calc(ps, ps, dt)
    ps.rx[...] = ps.mrx / mtot
    ps.ry[...] = ps.mry / mtot
    ps.rz[...] = ps.mrz / mtot
    U = 0.5 * ps.u.sum()
    type(ps).U = U
    type(ps).t_curr += dt
    return ps

#    ps.rx += dt * ps.vx
#    ps.ry += dt * ps.vy
#    ps.rz += dt * ps.vz
#    type(ps).U = -ps.potential_energy
#    type(ps).t_curr += dt
#    ps.set_acc(ps)
#    return ps


def nreg_v(ps, dt):
    """

    """
#    type(ps).W += 0.5 * dt * (ps.mass * (ps.vx * ps.ax
#                                         + ps.vy * ps.ay
#                                         + ps.vz * ps.az)).sum()
    mtot = ps.total_mass
    extensions.nreg_v.calc(ps, ps, dt)
    ps.vx[...] = ps.mvx / mtot
    ps.vy[...] = ps.mvy / mtot
    ps.vz[...] = ps.mvz / mtot
    K = 0.25 * ps.mk.sum() / mtot
    type(ps).W = (K - ps.E0)
#    type(ps).W += 0.5 * dt * (ps.mass * (ps.vx * ps.ax
#                                         + ps.vy * ps.ay
#                                         + ps.vz * ps.az)).sum()
    return ps

##    type(ps).W += 0.5 * dt * (ps.mass * (ps.vx * ps.ax
##                                         + ps.vy * ps.ay
##                                         + ps.vz * ps.az)).sum()
#    ps.vx += dt * ps.ax
#    ps.vy += dt * ps.ay
#    ps.vz += dt * ps.az
#    type(ps).W = (ps.kinetic_energy - ps.E0)
##    type(ps).W += 0.5 * dt * (ps.mass * (ps.vx * ps.ax
##                                         + ps.vy * ps.ay
##                                         + ps.vz * ps.az)).sum()
#    return ps


def anreg_step(ps, h):
    """

    """
    ps = nreg_x(ps, 0.5 * (h / ps.W))
    ps = nreg_v(ps, (h / ps.U))
    ps = nreg_x(ps, 0.5 * (h / ps.W))

    return ps


def nreg_step(ps, h):
    """

    """
    ps = anreg_step(ps, 0.5 * (h * ps.S))
    type(ps).S = 1/(2/ps.W - 1/ps.S)
    ps = anreg_step(ps, 0.5 * (h * ps.S))

    return ps


@decallmethods(timings)
class NREG(Base):
    """

    """
    PROVIDED_METHODS = ['nreg', 'anreg'
                        ]

    def __init__(self, eta, time, ps, method, **kwargs):
        """

        """
        super(NREG, self).__init__(eta, time, ps, **kwargs)
        self.method = method

    def initialize(self, t_end):
        """

        """
        logger.info("Initializing '%s' integrator.",
                    self.method)

        ps = self.ps
        type(ps).E0 = ps.kinetic_energy + ps.potential_energy
        type(ps).W = -ps.potential_energy
        type(ps).U = -ps.potential_energy
        type(ps).S = -ps.potential_energy

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

    def do_step(self, ps, tau):
        """

        """
        if "anreg" in self.method:
            t0 = ps.t_curr
            ps = anreg_step(ps, tau/2)
            t1 = ps.t_curr
        else:
            t0 = ps.t_curr
            ps = nreg_step(ps, tau)
            t1 = ps.t_curr

        dt = t1 - t0

        ps.tstep[...] = dt
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
