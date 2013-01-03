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


__all__ = ["BIOS"]

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
class LLBIOS(object):
    """

    """
    def __init__(self):
        self.kernel = kernels.bios_kernel
        self.kernel.local_size = 512
        self.drx = np.zeros(0, dtype=REAL)
        self.dry = np.zeros(0, dtype=REAL)
        self.drz = np.zeros(0, dtype=REAL)
        self.dvx = np.zeros(0, dtype=REAL)
        self.dvy = np.zeros(0, dtype=REAL)
        self.dvz = np.zeros(0, dtype=REAL)
        self.max_output_size = 0

        self.kernel.set_local_memory(26, 1)
        self.kernel.set_local_memory(27, 1)
        self.kernel.set_local_memory(28, 1)
        self.kernel.set_local_memory(29, 1)
        self.kernel.set_local_memory(30, 1)
        self.kernel.set_local_memory(31, 1)
        self.kernel.set_local_memory(32, 1)
        self.kernel.set_local_memory(33, 1)


    def set_args(self, iobj, jobj, dt):
        ni = iobj.n
        nj = jobj.n
        Mtot = jobj.total_mass

        if ni > self.max_output_size:
            self.drx = np.zeros(ni, dtype=REAL)
            self.dry = np.zeros(ni, dtype=REAL)
            self.drz = np.zeros(ni, dtype=REAL)
            self.dvx = np.zeros(ni, dtype=REAL)
            self.dvy = np.zeros(ni, dtype=REAL)
            self.dvz = np.zeros(ni, dtype=REAL)
            self.max_output_size = ni

        self.kernel.global_size = ni
        self.kernel.set_int(0, ni)
        self.kernel.set_input_buffer(1, iobj.x)
        self.kernel.set_input_buffer(2, iobj.y)
        self.kernel.set_input_buffer(3, iobj.z)
        self.kernel.set_input_buffer(4, iobj.mass)
        self.kernel.set_input_buffer(5, iobj.vx)
        self.kernel.set_input_buffer(6, iobj.vy)
        self.kernel.set_input_buffer(7, iobj.vz)
        self.kernel.set_input_buffer(8, iobj.eps2)
        self.kernel.set_int(9, nj)
        self.kernel.set_input_buffer(10, jobj.x)
        self.kernel.set_input_buffer(11, jobj.y)
        self.kernel.set_input_buffer(12, jobj.z)
        self.kernel.set_input_buffer(13, jobj.mass)
        self.kernel.set_input_buffer(14, jobj.vx)
        self.kernel.set_input_buffer(15, jobj.vy)
        self.kernel.set_input_buffer(16, jobj.vz)
        self.kernel.set_input_buffer(17, jobj.eps2)
        self.kernel.set_float(18, Mtot)
        self.kernel.set_float(19, dt)
        self.kernel.set_output_buffer(20, self.drx[:ni])
        self.kernel.set_output_buffer(21, self.dry[:ni])
        self.kernel.set_output_buffer(22, self.drz[:ni])
        self.kernel.set_output_buffer(23, self.dvx[:ni])
        self.kernel.set_output_buffer(24, self.dvy[:ni])
        self.kernel.set_output_buffer(25, self.dvz[:ni])


    def run(self):
        self.kernel.run()


    def get_result(self):
        return self.kernel.get_result()



llbios = LLBIOS()


def sakura(p, tau, update_com=False):
    llbios.set_args(p, p, tau)
    llbios.run()
    (dx, dy, dz, dvx, dvy, dvz) = llbios.get_result()

    if update_com:
        vcom = p.vcom
        dx += vcom[0] * tau
        dy += vcom[1] * tau
        dz += vcom[2] * tau

    p.x += dx
    p.y += dy
    p.z += dz
    p.vx += dvx
    p.vy += dvy
    p.vz += dvz

    return p


@decallmethods(timings)
class BIOS(Base):
    """

    """
    def __init__(self, eta, time, particles, **kwargs):
        super(BIOS, self).__init__(eta, time, particles, **kwargs)
        self.e0 = None


    def do_step(self, p, tau):
        """

        """
#        p0 = p.copy()
#        if self.e0 is None:
#            self.e0 = p0.kinetic_energy + p0.potential_energy
#        de = [1]
#        tol = tau**2
#        nsteps = 1
#
#        while abs(de[0]) > tol:
#            p = p0.copy()
#            dt = tau / nsteps
#            for i in range(nsteps):
#                p = sakura(p, dt)
#                e1 = p.kinetic_energy + p.potential_energy
#                de[0] = e1/self.e0 - 1
#                if abs(de[0]) > tol:
##                    nsteps += (nsteps+1)//2
#                    nsteps *= 2
##                    print(nsteps, de, tol)
#                    break


        p = sakura(p, tau)

        p.tstep = tau
        p.time += tau
        p.nstep += 1
        return p


    def get_base_tstep(self, t_end):
        self.tstep = self.eta
        if abs(self.time + self.tstep) > t_end:
            self.tstep = math.copysign(t_end - abs(self.time), self.eta)
        return self.tstep


    def initialize(self, t_end):
        logger.info("Initializing '%s' integrator.", type(self).__name__.lower())

        p = self.particles

        if self.dumpper:
            self.snap_number = 0
            self.dumpper.dump_snapshot(p, self.snap_number)

        self.is_initialized = True


    def finalize(self, t_end):
        logger.info("Finalizing '%s' integrator.", type(self).__name__.lower())

        p = self.particles
        tau = self.get_base_tstep(t_end)
        p.tstep = tau

        if self.reporter:
            self.reporter.report(self.time, p)


    def evolve_step(self, t_end):
        """

        """
        if not self.is_initialized:
            self.initialize(t_end)

        p = self.particles
        tau = self.get_base_tstep(t_end)

        p.tstep = tau

        if self.reporter:
            self.reporter.report(self.time, p)

        p = self.do_step(p, tau)
        self.time += tau

        if self.dumpper:
            pp = p[p.nstep % self.dump_freq == 0]
            if pp.n:
                self.snap_number += 1
                self.dumpper.dump_snapshot(pp, self.snap_number)

        self.particles = p


########## end of file ##########
