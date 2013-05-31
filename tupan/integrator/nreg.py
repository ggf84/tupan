# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function, division
import math
import logging
from ..integrator import Base
from ..lib.gravity import nreg_x as llnreg_x
from ..lib.gravity import nreg_v as llnreg_v
from ..lib.utils.timing import decallmethods, timings


__all__ = ["NREG"]


logger = logging.getLogger(__name__)


def nreg_x(p, t, dt):
#    llnreg_x.set_args(p, p, dt)
#    llnreg_x.run()
#    (rx, ry, rz, ax, ay, az, u) = llnreg_x.get_result()
#    U = 0.5 * u.sum()
#
#    mtot = p.total_mass
#
#    p.rx = rx / mtot
#    p.ry = ry / mtot
#    p.rz = rz / mtot
#
#    p.ax = ax.copy()
#    p.ay = ay.copy()
#    p.az = az.copy()
#
#    t += dt
#    return t, U

    p.rx += dt * p.vx
    p.ry += dt * p.vy
    p.rz += dt * p.vz
    (p.ax, p.ay, p.az) = p.get_acc(p)
    U = -p.potential_energy
    t += dt
    return t, U


def nreg_v(p, W, dt):
#    W += 0.5 * dt * (p.mass * (p.vx * p.ax
#                               + p.vy * p.ay
#                               + p.vz * p.az)).sum()
#
#    llnreg_v.set_args(p, p, dt)
#    llnreg_v.run()
#    (vx, vy, vz, k) = llnreg_v.get_result()
#    mtot = p.total_mass
#    p.vx = vx / mtot
#    p.vy = vy / mtot
#    p.vz = vz / mtot
#
#    W += 0.5 * dt * (p.mass * (p.vx * p.ax
#                               + p.vy * p.ay
#                               + p.vz * p.az)).sum()
#    return W

    W += 0.5 * dt * (p.mass * (p.vx * p.ax
                               + p.vy * p.ay
                               + p.vz * p.az)).sum()
    p.vx += dt * p.ax
    p.vy += dt * p.ay
    p.vz += dt * p.az
    W += 0.5 * dt * (p.mass * (p.vx * p.ax
                               + p.vy * p.ay
                               + p.vz * p.az)).sum()
    return W


def get_h(p, tau):
    W0 = -p.potential_energy
    h = tau * W0
    err = 1.0
    tol = 2.0**(-42)
    while err > tol:
        p0 = p.copy()
        p1, dt, W1 = nreg_base_step(p0, h)
        h = 2 * tau * (W0 * W1) / (W0 + W1)
        err0 = err
        err = abs((dt - tau) / tau)
        if err0 < err:
            return h, True
    return h, False


def nreg_base_step(p, h):
    t = 0.0
    W = -p.potential_energy

    t, U = nreg_x(p, t, 0.5 * (h / W))
    W = nreg_v(p, W, (h / U))
    t, U = nreg_x(p, t, 0.5 * (h / W))

    return p, t, W


def nreg_step(p, tau, nsteps=1):
    t = 0.0
    dtau = tau / nsteps
    for i in range(nsteps):
        h, err = get_h(p, dtau)
        if not err:
            p, dt, W = nreg_base_step(p, h)
        else:
            p, dt, W = nreg_step(p, dtau, 2*nsteps)
        t += dt
    return p, t, W


@decallmethods(timings)
class NREG(Base):
    """

    """
    def __init__(self, eta, time, particles, **kwargs):
        super(NREG, self).__init__(eta, time, particles, **kwargs)

    def do_step(self, p, tau):
        """

        """
        p, t, W = nreg_step(p, tau)
#        p, t, W = nreg_base_step(p, tau)   # h = tau

        p.tstep = t
        p.time += t
        p.nstep += 1
        return p, t

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

#        if self.dumpper:
#            self.snap_number = 0
#            self.dumpper.dump_snapshot(p, self.snap_number)
        self.is_initialized = True

    def finalize(self, t_end):
        logger.info(
            "Finalizing '%s' integrator.",
            type(self).__name__.lower()
        )

        p = self.particles
        tau = self.get_base_tstep(t_end)

        if self.reporter:
            self.reporter.report(self.time, p)

    def evolve_step(self, t_end):
        """

        """
        if not self.is_initialized:
            self.initialize(t_end)

        p = self.particles
        tau = self.get_base_tstep(t_end)

        if self.reporter:
            self.reporter.report(self.time, p)

        p, dt = self.do_step(p, tau)
        self.time += dt

#        if self.dumpper:
#            pp = p[p.nstep % self.dump_freq == 0]
#            if pp.n:
#                self.snap_number += 1
#                self.dumpper.dump_snapshot(pp, self.snap_number)

        self.particles = p


########## end of file ##########
