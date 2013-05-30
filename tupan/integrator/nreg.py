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
    (ax, ay, az) = p.get_acc(p)
    p.ax = ax.copy()
    p.ay = ay.copy()
    p.az = az.copy()
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


def get_h(p, tau, W):
    W0 = W
    h = tau * W0
    err = 1.0
    tol = 2.0**(-52)
    i = 0
    while err > tol:
        h0 = h
        p0 = p.copy()
        p1, dt, W1, U1 = nreg_step(p0, h0, W0)
        h = 2 * tau * (W0 * W1) / (W0 + W1)
        err = abs((dt - tau) / tau)
        i += 1
        if i > 32:
            return h, True
    return h, False


def nreg_step(p, h, W):
    t = 0.0

    t, U = nreg_x(p, t, 0.5 * (h / W))
    W = nreg_v(p, W, (h / U))
    t, U = nreg_x(p, t, 0.5 * (h / W))

    return p, t, W, U


@decallmethods(timings)
class NREG(Base):
    """

    """
    def __init__(self, eta, time, particles, **kwargs):
        super(NREG, self).__init__(eta, time, particles, **kwargs)
        self.U = None
        self.W = None

    def do_step(self, p, tau):
        """

        """
        def step(p, tau, W, nsteps=1):
            t = 0.0
            dtau = tau / nsteps
            for i in range(nsteps):
                h, err = get_h(p, dtau, W)
                if not err:
                    p, dt, W, U = nreg_step(p, h, W)
                else:
                    p, dt, W, U = step(p, dtau, W, 2*nsteps)
                t += dt
            return p, t, W, U

        W = self.W
        U = self.U
        p, t, W, U = step(p, tau, W)
#        p, t, W, U = nreg_step(p, tau, W)   # h = tau
        self.U = U
        self.W = W

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
        U = -p.potential_energy
        (ax, ay, az) = p.get_acc(p)
        p.ax = ax.copy()
        p.ay = ay.copy()
        p.az = az.copy()
        self.U = U
        self.W = U

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
