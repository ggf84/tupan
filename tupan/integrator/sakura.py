# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function, division
import logging
import math
from ..integrator import Base
from ..lib.gravity import sakura as llsakura
from ..lib.utils.timing import decallmethods, timings


__all__ = ["SAKURA"]


logger = logging.getLogger(__name__)


def sakura_step(p, tau):
    p.rx += p.vx * tau / 2
    p.ry += p.vy * tau / 2
    p.rz += p.vz * tau / 2

    llsakura.set_args(p, p, tau)
    llsakura.run()
    (drx, dry, drz, dvx, dvy, dvz) = llsakura.get_result()
    p.rx += drx
    p.ry += dry
    p.rz += drz
    p.vx += dvx
    p.vy += dvy
    p.vz += dvz

    p.rx += p.vx * tau / 2
    p.ry += p.vy * tau / 2
    p.rz += p.vz * tau / 2

    return p


@decallmethods(timings)
class SAKURA(Base):
    """

    """
    def __init__(self, eta, time, particles, **kwargs):
        super(SAKURA, self).__init__(eta, time, particles, **kwargs)
        self.e0 = None

    def get_sakura_tstep(self, isys, jsys, eta):
        (tstep_a, tstep_b) = isys.get_tstep(jsys, eta)

        iw2_a = (eta/tstep_a)**2
        iw2_b = (eta/tstep_b)**2

        diw2 = (iw2_a - iw2_b)

#        beta = 1
#        w2_sakura = ((beta+1)/(1/diw2 - beta/iw2_a)).max()
#        w2_sakura = (1/(1/diw2**beta - 1/iw2_a**beta)).max()**(1.0/beta)
#        w2_sakura = (1/(1/diw2**beta + 1/iw2_b**beta)).max()**(1.0/beta)

        w2_sakura = diw2.max()

        tstep_sakura = eta/w2_sakura**0.5

        return tstep_sakura

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
#                p = sakura_step(p, dt)
#                e1 = p.kinetic_energy + p.potential_energy
#                de[0] = e1/self.e0 - 1
#                if abs(de[0]) > tol:
##                    nsteps += (nsteps+1)//2
#                    nsteps *= 2
##                    print(nsteps, de, tol)
#                    break

        p = sakura_step(p, tau)

        p.tstep = tau
        p.time += tau
        p.nstep += 1
        return p

    def get_base_tstep(self, t_end):
        p = self.particles
        self.tstep = self.get_sakura_tstep(p, p, self.eta)
#        self.tstep = self.eta
        if abs(self.time + self.tstep) > t_end:
            self.tstep = math.copysign(t_end - abs(self.time), self.eta)
        return self.tstep

    def initialize(self, t_end):
        logger.info(
            "Initializing '%s' integrator.",
            type(self).__name__.lower()
        )

        p = self.particles

        if self.dumpper:
            self.snap_number = 0
            self.dumpper.dump_snapshot(p, self.snap_number)

        self.is_initialized = True

    def finalize(self, t_end):
        logger.info(
            "Finalizing '%s' integrator.",
            type(self).__name__.lower()
        )

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
