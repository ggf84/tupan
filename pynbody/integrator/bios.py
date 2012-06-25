#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import print_function
import logging
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
            raise TypeError(msg.format(self.__class__.__name__,", ".join(kwargs.keys())))



@decallmethods(timings)
class AbstractBIOS(Base):
    """

    """
    def __init__(self, eta, time, particles, **kwargs):
        super(AbstractBIOS, self).__init__(eta, time, particles, **kwargs)
        self.kernel = kernels.bios_kernel
        self.kernel.local_size = 384
        self.kernel.set_arg('LMEM', 5, 8)
        self.fields = ('pos', 'mass', 'vel', 'eps2')

    def set_args(self, iobj, jobj, dt):
        ni = iobj.n
        nj = jobj.n
        idata = iobj.stack_fields(self.fields)
        jdata = jobj.stack_fields(self.fields)
        self.kernel.set_arg('IN', 0, ni)
        self.kernel.set_arg('IN', 1, idata)
        self.kernel.set_arg('IN', 2, nj)
        self.kernel.set_arg('IN', 3, jdata)
        self.kernel.set_arg('IN', 4, dt)
        self.kernel.set_arg('OUT', 5, (ni, 8))
        self.kernel.global_size = ni

    def run(self):
        self.kernel.run()

    def get_result(self):
        result = self.kernel.get_result()[0]
        return (result[:,:3], result[:,4:7])



@decallmethods(timings)
class BIOS(AbstractBIOS):
    """

    """
    def __init__(self, eta, time, particles, **kwargs):
        super(BIOS, self).__init__(eta, time, particles, **kwargs)


    def do_step(self, p, tau):
        """

        """
        self.set_args(p, p, tau)
        self.run()
        (dr, dv) = self.get_result()
        for iobj in p.values():
            if iobj.n:
                iobj.pos += dr[:iobj.n]
                iobj.vel += dv[:iobj.n]
                dr = dr[iobj.n:]
                dv = dv[iobj.n:]

        p.set_dt_prev(tau)
        p.update_t_curr(tau)
        p.update_nstep()
        return p


    def get_base_tstep(self, t_end):
        tau = self.eta
        self.tstep = tau if self.time + tau <= t_end else t_end - self.time
        return self.tstep


    def initialize(self, t_end):
        logger.info("Initializing '%s' integrator.", self.__class__.__name__.lower())

        p = self.particles

        p.update_acc(p)
        if self.pn_order > 0:
            p.update_pnacc(p, self.pn_order, self.clight)

        self.is_initialized = True


    def finalize(self, t_end):
        logger.info("Finalizing '%s' integrator.", self.__class__.__name__.lower())

        p = self.particles
        tau = self.get_base_tstep(t_end)
        p.set_dt_next(tau)

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
        p.set_dt_next(tau)

        if self.reporter:
            self.reporter.report(self.time, p)
        if self.dumpper:
            self.dumpper.dump(p.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        p = self.do_step(p, tau)
        self.time += tau

        self.particles = p


########## end of file ##########
