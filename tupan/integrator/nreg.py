#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import print_function, division
import logging
import math
import numpy as np
from ..lib.extensions import kernels
from ..lib.utils.timing import decallmethods, timings
from ..lib.utils.dtype import *


__all__ = ["NREG"]

logger = logging.getLogger(__name__)


class Base(object):
    """

    """
    def __init__(self, eta, time, particles, **kwargs):
        self.eta = eta
        self.time = time
        self.particles = particles
        self.is_initialized = False

        self.pn_order = kwargs.pop("pn_order", 0)
        self.clight = kwargs.pop("clight", None)
        if self.pn_order > 0 and self.clight is None:
            raise TypeError("'clight' is not defined. Please set the speed of "
                            "light argument 'clight' when using 'pn_order' > 0.")

        self.reporter = kwargs.pop("reporter", None)
        self.viewer = kwargs.pop("viewer", None)
        self.dumpper = kwargs.pop("dumpper", None)
        self.dump_freq = kwargs.pop("dump_freq", 1)
        if kwargs:
            msg = "{0}.__init__ received unexpected keyword arguments: {1}."
            raise TypeError(msg.format(type(self).__name__,", ".join(kwargs.keys())))



@decallmethods(timings)
class LLNREG_X(object):
    """

    """
    def __init__(self, libnreg):
        self.kernel = libnreg.nreg_Xkernel
        self.kernel.local_size = 512
        self.rx = np.zeros(0, dtype=REAL)
        self.ry = np.zeros(0, dtype=REAL)
        self.rz = np.zeros(0, dtype=REAL)
        self.ax = np.zeros(0, dtype=REAL)
        self.ay = np.zeros(0, dtype=REAL)
        self.az = np.zeros(0, dtype=REAL)
        self.u = np.zeros(0, dtype=REAL)
        self.max_output_size = 0

#        self.kernel.set_local_memory(26, 1)
#        self.kernel.set_local_memory(27, 1)
#        self.kernel.set_local_memory(28, 1)
#        self.kernel.set_local_memory(29, 1)
#        self.kernel.set_local_memory(30, 1)
#        self.kernel.set_local_memory(31, 1)
#        self.kernel.set_local_memory(32, 1)
#        self.kernel.set_local_memory(33, 1)


    def set_args(self, iobj, jobj, dt):
        ni = iobj.n
        nj = jobj.n

        self.osize = ni
        if ni > self.max_output_size:
            self.rx = np.zeros(ni, dtype=REAL)
            self.ry = np.zeros(ni, dtype=REAL)
            self.rz = np.zeros(ni, dtype=REAL)
            self.ax = np.zeros(ni, dtype=REAL)
            self.ay = np.zeros(ni, dtype=REAL)
            self.az = np.zeros(ni, dtype=REAL)
            self.u = np.zeros(ni, dtype=REAL)
            self.max_output_size = ni

        self.kernel.global_size = ni
        self.kernel.set_int(0, ni)
        self.kernel.set_array(1, iobj.x)
        self.kernel.set_array(2, iobj.y)
        self.kernel.set_array(3, iobj.z)
        self.kernel.set_array(4, iobj.mass)
        self.kernel.set_array(5, iobj.vx)
        self.kernel.set_array(6, iobj.vy)
        self.kernel.set_array(7, iobj.vz)
        self.kernel.set_array(8, iobj.eps2)
        self.kernel.set_int(9, nj)
        self.kernel.set_array(10, jobj.x)
        self.kernel.set_array(11, jobj.y)
        self.kernel.set_array(12, jobj.z)
        self.kernel.set_array(13, jobj.mass)
        self.kernel.set_array(14, jobj.vx)
        self.kernel.set_array(15, jobj.vy)
        self.kernel.set_array(16, jobj.vz)
        self.kernel.set_array(17, jobj.eps2)
        self.kernel.set_float(18, dt)
        self.kernel.set_output_buffer(19, self.rx)
        self.kernel.set_output_buffer(20, self.ry)
        self.kernel.set_output_buffer(21, self.rz)
        self.kernel.set_output_buffer(22, self.ax)
        self.kernel.set_output_buffer(23, self.ay)
        self.kernel.set_output_buffer(24, self.az)
        self.kernel.set_output_buffer(25, self.u)


    def run(self):
        self.kernel.run()


    def get_result(self):
        ni = self.osize
        return [self.rx[:ni], self.ry[:ni], self.rz[:ni],
                self.ax[:ni], self.ay[:ni], self.az[:ni],
                0.5 * self.u[:ni].sum()]


@decallmethods(timings)
class LLNREG_V(object):
    """

    """
    def __init__(self, libnreg):
        self.kernel = libnreg.nreg_Vkernel
        self.kernel.local_size = 512
        self.vx = np.zeros(0, dtype=REAL)
        self.vy = np.zeros(0, dtype=REAL)
        self.vz = np.zeros(0, dtype=REAL)
        self.k = np.zeros(0, dtype=REAL)
        self.max_output_size = 0

#        self.kernel.set_local_memory(21, 1)
#        self.kernel.set_local_memory(22, 1)
#        self.kernel.set_local_memory(23, 1)
#        self.kernel.set_local_memory(24, 1)
#        self.kernel.set_local_memory(25, 1)
#        self.kernel.set_local_memory(26, 1)
#        self.kernel.set_local_memory(27, 1)


    def set_args(self, iobj, jobj, dt):
        ni = iobj.n
        nj = jobj.n

        self.osize = ni
        if ni > self.max_output_size:
            self.vx = np.zeros(ni, dtype=REAL)
            self.vy = np.zeros(ni, dtype=REAL)
            self.vz = np.zeros(ni, dtype=REAL)
            self.k = np.zeros(ni, dtype=REAL)
            self.max_output_size = ni

        self.kernel.global_size = ni
        self.kernel.set_int(0, ni)
        self.kernel.set_array(1, iobj.vx)
        self.kernel.set_array(2, iobj.vy)
        self.kernel.set_array(3, iobj.vz)
        self.kernel.set_array(4, iobj.mass)
        self.kernel.set_array(5, iobj.ax)
        self.kernel.set_array(6, iobj.ay)
        self.kernel.set_array(7, iobj.az)
        self.kernel.set_int(8, nj)
        self.kernel.set_array(9, jobj.vx)
        self.kernel.set_array(10, jobj.vy)
        self.kernel.set_array(11, jobj.vz)
        self.kernel.set_array(12, jobj.mass)
        self.kernel.set_array(13, jobj.ax)
        self.kernel.set_array(14, jobj.ay)
        self.kernel.set_array(15, jobj.az)
        self.kernel.set_float(16, dt)
        self.kernel.set_output_buffer(17, self.vx)
        self.kernel.set_output_buffer(18, self.vy)
        self.kernel.set_output_buffer(19, self.vz)
        self.kernel.set_output_buffer(20, self.k)


    def run(self):
        self.kernel.run()


    def get_result(self):
        ni = self.osize
        return [self.vx[:ni], self.vy[:ni], self.vz[:ni],
                0.5 * self.k[:ni].sum()]


llnreg_x = LLNREG_X(kernels)
llnreg_v = LLNREG_V(kernels)


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

    W += 0.5 * h * (p.mass * (p.vx * p.ax + p.vy * p.ay + p.vz * p.az)).sum() / U

    K = nreg_v(p, h/U)

    W += 0.5 * h * (p.mass * (p.vx * p.ax + p.vy * p.ay + p.vz * p.az)).sum() / U

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


    def do_step(self, p, tau):
        """

        """
        def do_nsteps(p, tau, W, U, nsteps):
            t = 0.0
            dtau = tau / nsteps
            for i in range(nsteps):
                S = W + 0.5 * dtau * (p.mass * (p.vx * p.ax + p.vy * p.ay + p.vz * p.az)).sum()
                tt, W, U = nreg_step(p, S*dtau, W, U)
                t += tt
            return p, t, W, U


#        i = 1
#        tol = max(min(2.0**(-12), tau**2), 2.0**(-32))
#        while True:
#            U = self.U
#            W = self.W
#            p0 = p.copy()
#            p0, t, W, U = do_nsteps(p0, tau, W, U, i)
#            i *= 2
#            if abs(W-U)/U < tol:
#                if abs(t-tau)/tau < tol:
#                    break
#
#        self.W = W
#        self.U = U
#        p = p0.copy()



        U = self.U
        W = self.W
        p, t, W, U = do_nsteps(p, tau, W, U, 1)
        self.W = W
        self.U = U

#        print(self.W, self.U, abs(self.W-self.U))

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
        logger.info("Initializing '%s' integrator.", type(self).__name__.lower())

        p = self.particles
        U = nreg_x(p, 0.0)
        K = nreg_v(p, 0.0)
        self.U = U
        self.W = U


#        if self.dumpper:
#            self.snap_number = 0
#            self.dumpper.dump_snapshot(p, self.snap_number)

        self.is_initialized = True


    def finalize(self, t_end):
        logger.info("Finalizing '%s' integrator.", type(self).__name__.lower())

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
