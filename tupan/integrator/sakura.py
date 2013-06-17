# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function, division
import logging
from ..integrator import Base
from ..lib.gravity import sakura as llsakura
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Sakura"]


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
class Sakura(Base):
    """

    """
    PROVIDED_METHODS = ['sakura', 'asakura',
                        ]

    def __init__(self, eta, time, ps, method, **kwargs):
        """

        """
        super(Sakura, self).__init__(eta, time, ps, **kwargs)
        self.method = method
        self.e0 = None

    def initialize(self, t_end):
        """

        """
        logger.info("Initializing '%s' integrator.",
                    self.method)

        ps = self.ps

        if self.reporter:
            self.reporter.diagnostic_report(self.time, ps)
        if self.dumpper:
            self.dumpper.dump_worldline(ps)

        self.is_initialized = True

    def finalize(self, t_end):
        """

        """
        logger.info("Finalizing '%s' integrator.",
                    self.method)

    def get_sakura_tstep(self, ps, eta, tau):
        """

        """
        (tstep_a, tstep_b) = ps.get_tstep(ps, eta)

        iw2_a = (eta/tstep_a)**2
        iw2_b = (eta/tstep_b)**2

        diw2 = (iw2_a - iw2_b)

        w2_sakura = diw2.max()
        dt_sakura = eta/(1 + w2_sakura)**0.5

        if abs(dt_sakura) > abs(tau):
            dt_sakura = tau

        self.tstep = dt_sakura
        return self.tstep

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

        if "asakura" in self.method:
            tau = self.get_sakura_tstep(ps, self.eta, tau)
        ps = sakura_step(ps, tau)
        self.time += tau

        ps.tstep = tau
        ps.time += tau
        ps.nstep += 1
        wp = ps[ps.time % (self.dump_freq * tau) == 0]
        if wp.n:
            self.wl.append(wp.copy())
        return ps


########## end of file ##########
