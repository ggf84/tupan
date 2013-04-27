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
    (rx, ry, rz, ax, ay, az, U) = llnreg_x.get_result()

    mtot = p.total_mass

    p.x = rx / mtot
    p.y = ry / mtot
    p.z = rz / mtot

    p.ax = ax.copy()
    p.ay = ay.copy()
    p.az = az.copy()

#    p.x += dt * p.vx
#    p.y += dt * p.vy
#    p.z += dt * p.vz
#    (ax, ay, az) = p.get_acc(p)
#    U = -p.potential_energy
    return U


def nreg_v(p, dt):
    llnreg_v.set_args(p, p, dt)
    llnreg_v.run()
    (vx, vy, vz, K) = llnreg_v.get_result()

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

    dt = 0.5 * h / W
    t += dt
    U = nreg_x(p, dt)

    W += 0.5 * h * (p.mass * (
        p.vx * p.ax + p.vy * p.ay + p.vz * p.az)).sum() / U

    K = nreg_v(p, h/U)

    W += 0.5 * h * (p.mass * (
        p.vx * p.ax + p.vy * p.ay + p.vz * p.az)).sum() / U

    dt = 0.5 * h / W
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
        self.S = None

    def do_step(self, p, tau):
        """

        """
        def do_nsteps(p, tau, W, U, S, nsteps, gamma=0.5):
            t = 0.0
            h = tau / nsteps
            for i in range(nsteps):
                S += 0.5 * (h/(abs(W)**gamma)) * (p.mass * (
                    p.vx * p.ax + p.vy * p.ay + p.vz * p.az)).sum() / U
                tt, W, U = nreg_step(p, h/(abs(S)**gamma), W, U)
                S += 0.5 * (h/(abs(W)**gamma)) * (p.mass * (
                    p.vx * p.ax + p.vy * p.ay + p.vz * p.az)).sum() / U
                t += tt
            return p, t, W, U, S

#        i = 1
#        tol = 1.0e-2#max(min(2.0**(-12), tau**2), 2.0**(-32))
#        while True:
#            U = self.U
#            W = self.W
#            S = self.S
#            p0 = p.copy()
#            p0, t, W, U, S = do_nsteps(p0, tau, W, U, S, i)
#            i *= 2
#            if abs(W-U)/U < tol:
##                if abs(t-tau)/tau < tol:
#                break
#
#        self.S = S
#        self.W = W
#        self.U = U
#        p = p0.copy()
        U = self.U
        W = self.W
        S = self.S
        p, t, W, U, S = do_nsteps(p, tau, W, U, S, 64)
        self.S = S
        self.W = W
        self.U = U

#        print(self.W, self.U, self.S, abs(self.W-self.U))

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
        K = nreg_v(p, 0.0)
        self.U = U
        self.W = U
        self.S = U

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
