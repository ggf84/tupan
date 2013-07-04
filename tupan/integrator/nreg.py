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


def nreg_x(ps, t, dt):
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
    (ps.ax, ps.ay, ps.az) = ps.get_acc(ps)
    U = -ps.potential_energy
    t += dt
    return t, U


def nreg_v(ps, W, dt):
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

    W += 0.5 * dt * (ps.mass * (ps.vx * ps.ax
                                + ps.vy * ps.ay
                                + ps.vz * ps.az)).sum()
    ps.vx += dt * ps.ax
    ps.vy += dt * ps.ay
    ps.vz += dt * ps.az
    W += 0.5 * dt * (ps.mass * (ps.vx * ps.ax
                                + ps.vy * ps.ay
                                + ps.vz * ps.az)).sum()
    return W


def get_h(ps, tau):
    """

    """
    W0 = -ps.potential_energy
    h = tau * W0
    err = 1.0
    tol = 2.0**(-48)
    while err > tol:
        ps1, dt, W1 = nreg_base_step(ps.copy(), h)
        h = 2 * tau * (W0 * W1) / (W0 + W1)
        err0 = err
        err = abs((dt - tau) / tau)
        if err0 < err:
            return h, True
    return h, False


def nreg_base_step(ps, h):
    """

    """
    t = 0.0
    W = -ps.potential_energy

    t, U = nreg_x(ps, t, 0.5 * (h / W))
    W = nreg_v(ps, W, (h / U))
    t, U = nreg_x(ps, t, 0.5 * (h / W))

    return ps, t, W


def nreg_step(ps, tau, nsteps=1):
    """

    """
    t = 0.0
    dtau = tau / nsteps
    for i in range(nsteps):
        h, err = get_h(ps, dtau)
        if not err:
            ps, dt, W = nreg_base_step(ps, h)
        else:
            ps, dt, W = nreg_step(ps, dtau, 2*nsteps)
        t += dt
    return ps, t, W


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
        (ps.ax, ps.ay, ps.az) = ps.get_acc(ps)

        if self.reporter:
            self.reporter.diagnostic_report(ps)
        if self.dumpper:
            self.dumpper.dump_worldline(ps)

        self.is_initialized = True

    def finalize(self, t_end):
        """

        """
        logger.info("Finalizing '%s' integrator.",
                    self.method)

    def do_step(self, ps, tau):
        """

        """
#        ps, t, W = nreg_step(ps, tau)
        ps, t, W = nreg_base_step(ps, tau)   # h = tau
        tau = t

        type(ps).t_curr += tau
        ps.tstep = tau
        ps.time += tau
        ps.nstep += 1
        wp = ps[ps.time % (self.dump_freq * tau) == 0]
        if wp.n:
            self.wl.append(wp.copy())
        return ps


########## end of file ##########
