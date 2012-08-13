#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import print_function
import logging
import math
import numpy as np
from ..lib.extensions import kernels
from ..lib.utils.timing import decallmethods, timings


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
        self.kernel.local_size = 384
        self.output = np.zeros((0, 8), dtype=self.kernel.env.dtype)


    def set_args(self, iobj, jobj, dt):
        ni = iobj.n
        nj = jobj.n
        idata = np.concatenate((iobj.pos, iobj.mass.reshape(-1,1),
                                iobj.vel, iobj.eps2.reshape(-1,1)), axis=1)
        jdata = np.concatenate((jobj.pos, jobj.mass.reshape(-1,1),
                                jobj.vel, jobj.eps2.reshape(-1,1)), axis=1)

        if ni > len(self.output):
            self.output = np.zeros((ni, 8), dtype=self.kernel.env.dtype)

        self.kernel.global_size = ni
        self.kernel.set_int(0, ni)
        self.kernel.set_input_buffer(1, idata)
        self.kernel.set_int(2, nj)
        self.kernel.set_input_buffer(3, jdata)
        self.kernel.set_float(4, dt)
        self.kernel.set_output_buffer(5, self.output[:ni])
        self.kernel.set_local_memory(6, 8)


    def run(self):
        self.kernel.run()


    def get_result(self):
        result = self.kernel.get_result()[0]
        return (result[:,:3], result[:,4:7])



llbios = LLBIOS()


@decallmethods(timings)
class BIOS(Base):
    """

    """
    def __init__(self, eta, time, particles, **kwargs):
        super(BIOS, self).__init__(eta, time, particles, **kwargs)


    def do_step(self, p, tau):
        """

        """
        vcm = p.get_center_of_mass_velocity()

        llbios.set_args(p, p, tau)
        llbios.run()
        (dr, dv) = llbios.get_result()

        p.pos += dr + tau * (p.vel - vcm)
        p.vel += dv

        p.dt_prev[:] = tau
        p.t_curr += tau
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

        p.update_acc(p)
        if self.pn_order > 0: p.update_pnacc(p, self.pn_order, self.clight)

        self.is_initialized = True


    def finalize(self, t_end):
        logger.info("Finalizing '%s' integrator.", type(self).__name__.lower())

        p = self.particles
        tau = self.get_base_tstep(t_end)
        p.dt_next[:] = tau

        if self.reporter:
            self.reporter.report(self.time, p)
        if self.dumpper:
            self.dumpper.dump(p)


    def evolve_step(self, t_end):
        """

        """
        if not self.is_initialized:
            self.initialize(t_end)

        p = self.particles
        tau = self.get_base_tstep(t_end)

        p.dt_next[:] = tau

        if self.reporter:
            self.reporter.report(self.time, p)
        if self.dumpper:
            self.dumpper.dump(p.select(p.nstep % self.dump_freq == 0))

        p = self.do_step(p, tau)
        self.time += tau

        self.particles = p


########## end of file ##########
