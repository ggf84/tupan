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


def nreg_x(p, dt):
    llnreg_x.set_args(p, p, dt)
    llnreg_x.run()
    (rx, ry, rz, ax, ay, az, u) = llnreg_x.get_result()
    U = 0.5 * u.sum()

    mtot = p.total_mass

    p.rx = rx / mtot
    p.ry = ry / mtot
    p.rz = rz / mtot

    p.ax = ax.copy()
    p.ay = ay.copy()
    p.az = az.copy()

#    p.rx += dt * p.vx
#    p.ry += dt * p.vy
#    p.rz += dt * p.vz
#    (ax, ay, az) = p.get_acc(p)
#    U = -p.potential_energy
    return U


def nreg_v(p, dt):
    llnreg_v.set_args(p, p, dt)
    llnreg_v.run()
    (vx, vy, vz, k) = llnreg_v.get_result()
    K = 0.5 * k.sum()

    mtot = p.total_mass

    p.vx = vx / mtot
    p.vy = vy / mtot
    p.vz = vz / mtot

    return K / mtot


#    p.vx += dt * p.ax
#    p.vy += dt * p.ay
#    p.vz += dt * p.az
#    K = p.kinetic_energy
#
#    return K


def nreg_step(p, h, W, U):
    t = 0.0

    dt = 0.5 * (h / W)
    t += dt
    U = nreg_x(p, dt)

    W += 0.5 * (h / U) * (p.mass * (p.vx * p.ax
                                    + p.vy * p.ay
                                    + p.vz * p.az)).sum()

    nreg_v(p, (h / U))

    W += 0.5 * (h / U) * (p.mass * (p.vx * p.ax
                                    + p.vy * p.ay
                                    + p.vz * p.az)).sum()

    dt = 0.5 * (h / W)
    U = nreg_x(p, dt)
    t += dt

    return t, W, U


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
        def do_nsteps(p, tau, W, U, nsteps):
            t = 0.0
            h = tau / nsteps
            for i in range(nsteps):
                dW = (h / U) * (p.mass * (p.vx * p.ax
                                          + p.vy * p.ay
                                          + p.vz * p.az)).sum()
                tt, W, U = nreg_step(p, h*W*(1+0.5*dW), W, U)
                t += tt
            return p, t, W, U

        i = 16
        tol = max((tau/4)**3, (2.0**(-53))**0.5)
        while True:
            W = self.W
            U = self.U
            p0 = p.copy()
            p0, t, W, U = do_nsteps(p0, tau, W, U, i)
            i *= 2
            if abs(W-U)/U < tol:
                if abs(t-tau)/tau < tol:
                    break
        self.U = U
        self.W = W
        p = p0.copy()

#        W = self.W
#        U = self.U
#        p, t, W, U = do_nsteps(p, tau, W, U, 16)
#        self.U = U
#        self.W = W

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
        U = nreg_x(p, 0.0)
        nreg_v(p, 0.0)
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
#        tau = self.get_base_tstep(t_end)
#        p.tstep = tau

        if self.reporter:
            self.reporter.report(self.time, p)

    def evolve_step(self, t_end):
        """

        """
        if not self.is_initialized:
            self.initialize(t_end)

        p = self.particles
#        tau = self.get_base_tstep(t_end)

#        p.tstep = tau

        if self.reporter:
            self.reporter.report(self.time, p)

        tau = self.eta
        p, dt = self.do_step(p, tau)
        self.time += dt

#        if self.dumpper:
#            pp = p[p.nstep % self.dump_freq == 0]
#            if pp.n:
#                self.snap_number += 1
#                self.dumpper.dump_snapshot(pp, self.snap_number)

        self.particles = p


########## end of file ##########
