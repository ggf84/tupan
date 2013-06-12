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


def sakura_step(ps, tau):
    """

    """
    ps.rx += ps.vx * tau / 2
    ps.ry += ps.vy * tau / 2
    ps.rz += ps.vz * tau / 2

    llsakura.set_args(ps, ps, tau)
    llsakura.run()
    (drx, dry, drz, dvx, dvy, dvz) = llsakura.get_result()
    ps.rx += drx
    ps.ry += dry
    ps.rz += drz
    ps.vx += dvx
    ps.vy += dvy
    ps.vz += dvz

    ps.rx += ps.vx * tau / 2
    ps.ry += ps.vy * tau / 2
    ps.rz += ps.vz * tau / 2

    return ps


@decallmethods(timings)
class SAKURA(Base):
    """

    """
    def __init__(self, eta, time, ps, **kwargs):
        """

        """
        super(SAKURA, self).__init__(eta, time, ps, **kwargs)
        self.e0 = None

    def get_base_tstep(self, t_end):
        """

        """
        ps = self.ps
        self.tstep = self.get_sakura_tstep(ps, ps, self.eta)
#        self.tstep = self.eta
        if abs(self.time + self.tstep) > t_end:
            self.tstep = math.copysign(t_end - abs(self.time), self.eta)
        return self.tstep

    def initialize(self, t_end):
        """

        """
        logger.info(
            "Initializing '%s' integrator.",
            type(self).__name__.lower()
        )

        ps = self.ps

        if self.reporter:
            self.reporter.diagnostic_report(self.time, ps)
        if self.dumpper:
            self.dumpper.dump_worldline(ps)

        self.is_initialized = True

    def finalize(self, t_end):
        """

        """
        logger.info(
            "Finalizing '%s' integrator.",
            type(self).__name__.lower()
        )

    def get_sakura_tstep(self, ips, jps, eta):
        """

        """
        (tstep_a, tstep_b) = ips.get_tstep(jps, eta)

        iw2_a = (eta/tstep_a)**2
        iw2_b = (eta/tstep_b)**2

        diw2 = (iw2_a - iw2_b)

        w2_sakura = diw2.max()
        dt_sakura = eta/(1 + w2_sakura)**0.5

        return dt_sakura

    def do_step(self, ps, tau):
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

        ps = sakura_step(ps, tau)

        ps.tstep = tau
        ps.time += tau
        ps.nstep += 1
        wp = ps[ps.nstep % self.dump_freq == 0]
        if wp.n:
            self.wl.append(wp.copy())
        return ps

    def evolve_step(self, t_end):
        """

        """
        if not self.is_initialized:
            self.initialize(t_end)

        ps = self.ps
        tau = self.get_base_tstep(t_end)
        self.wl = ps[:0]

        ps = self.do_step(ps, tau)

        self.time += tau
        self.ps = ps

        if self.reporter:
            self.reporter.diagnostic_report(self.time, ps)
        if self.dumpper:
            self.dumpper.dump_worldline(self.wl)


########## end of file ##########
