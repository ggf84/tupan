# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function, division
import logging
from ..integrator import Base
from ..lib.gravity import nreg_x as llnreg_x
from ..lib.gravity import nreg_v as llnreg_v
from ..lib.utils.timing import decallmethods, timings


__all__ = ["NREG"]


logger = logging.getLogger(__name__)


def nreg_x(ps, dt):
    """

    """
#    llnreg_x.set_args(ps, ps, dt)
#    llnreg_x.run()
#    (rx, ry, rz, ax, ay, az, u) = llnreg_x.get_result()
#    U = 0.5 * u.sum()
#
#    mtot = ps.total_mass
#
#    ps.rx = rx / mtot
#    ps.ry = ry / mtot
#    ps.rz = rz / mtot
#
#    ps.ax = ax.copy()
#    ps.ay = ay.copy()
#    ps.az = az.copy()
#
#    t += dt
#    return t, U

    ps.rx += dt * ps.vx
    ps.ry += dt * ps.vy
    ps.rz += dt * ps.vz
    type(ps).t_curr += dt
    type(ps).U = -ps.potential_energy
    (ps.ax, ps.ay, ps.az) = ps.get_acc(ps)
    return ps


def nreg_v(ps, dt):
    """

    """
#    W += 0.5 * dt * (ps.mass * (ps.vx * ps.ax
#                                + ps.vy * ps.ay
#                                + ps.vz * ps.az)).sum()
#
#    llnreg_v.set_args(ps, ps, dt)
#    llnreg_v.run()
#    (vx, vy, vz, k) = llnreg_v.get_result()
#    mtot = ps.total_mass
#    ps.vx = vx / mtot
#    ps.vy = vy / mtot
#    ps.vz = vz / mtot
#
#    W += 0.5 * dt * (ps.mass * (ps.vx * ps.ax
#                                + ps.vy * ps.ay
#                                + ps.vz * ps.az)).sum()
#    return W

    type(ps).W += 0.5 * dt * (ps.mass * (ps.vx * ps.ax
                                         + ps.vy * ps.ay
                                         + ps.vz * ps.az)).sum()
    ps.vx += dt * ps.ax
    ps.vy += dt * ps.ay
    ps.vz += dt * ps.az
    type(ps).W += 0.5 * dt * (ps.mass * (ps.vx * ps.ax
                                         + ps.vy * ps.ay
                                         + ps.vz * ps.az)).sum()
    return ps


def nreg_step(ps, h):
    """

    """
    ps = nreg_x(ps, 0.5 * (h / ps.W))
    ps = nreg_v(ps, (h / ps.U))
    ps = nreg_x(ps, 0.5 * (h / ps.W))

    return ps


def nreg_step2(ps, h):
    """

    """
    ps = nreg_x(ps, 0.5 * (h * ps.S / ps.W))
    ps = nreg_v(ps, (h * ps.S / ps.U))
    ps = nreg_x(ps, 0.5 * (h * ps.S / ps.W))
    type(ps).S = 1/(2/ps.W - 1/ps.S)
    ps = nreg_x(ps, 0.5 * (h * ps.S / ps.W))
    ps = nreg_v(ps, (h * ps.S / ps.U))
    ps = nreg_x(ps, 0.5 * (h * ps.S / ps.W))

    return ps


@decallmethods(timings)
class NREG(Base):
    """

    """
    PROVIDED_METHODS = ['nreg',
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
        type(ps).W = -ps.potential_energy
        type(ps).U = -ps.potential_energy
        type(ps).S = -ps.potential_energy
        (ps.ax, ps.ay, ps.az) = ps.get_acc(ps)

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
#        t0 = ps.t_curr
#        ps = nreg_step(ps, tau)
#        t1 = ps.t_curr

        t0 = ps.t_curr
        ps = nreg_step2(ps, tau/2)
        t1 = ps.t_curr

        dt = t1 - t0

        ps.tstep[:] = dt
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
