# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function, division
import logging
from ..integrator import Base
from ..lib import extensions
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Sakura"]


logger = logging.getLogger(__name__)


def sakura_step(ps, tau):
    """

    """
    ps.rx += ps.vx * tau / 2
    ps.ry += ps.vy * tau / 2
    ps.rz += ps.vz * tau / 2

    extensions.sakura.calc(ps, ps, tau/2, -1)
    ps.rx += ps.drx
    ps.ry += ps.dry
    ps.rz += ps.drz
    ps.vx += ps.dvx
    ps.vy += ps.dvy
    ps.vz += ps.dvz

    extensions.sakura.calc(ps, ps, tau/2, 1)
    ps.rx += ps.drx
    ps.ry += ps.dry
    ps.rz += ps.drz
    ps.vx += ps.dvx
    ps.vy += ps.dvy
    ps.vz += ps.dvz

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
            self.reporter.diagnostic_report(ps)
        if self.dumpper:
            self.dumpper.dump_worldline(ps)
        if self.viewer:
            self.viewer.show_event(ps)

        self.is_initialized = True

    def finalize(self, t_end):
        """

        """
        logger.info("Finalizing '%s' integrator.",
                    self.method)

        ps = self.ps

        if self.viewer:
            self.viewer.show_event(ps)
            self.viewer.enter_main_loop()

    def get_sakura_tstep(self, ps, eta, tau):
        """

        """
        ps.set_tstep(ps, eta)

        iw2_a = (eta/ps.tstep)**2
        iw2_b = (eta/ps.tstepij)**2

        diw2 = (iw2_a - iw2_b)

        w2_sakura = diw2.max()
        dt_sakura = eta/(1 + w2_sakura)**0.5

        ps.tstep[...] = dt_sakura

        min_bts = self.get_min_block_tstep(ps, tau)
        return min_bts

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

        type(ps).t_curr += tau
        ps.tstep[...] = tau
        ps.time += tau
        ps.nstep += 1
        if self.dumpper:
            slc = ps.time % (self.dump_freq * tau) == 0
            if any(slc):
                self.wl.append(ps[slc])
        if self.viewer:
            slc = ps.time % (self.gl_freq * tau) == 0
            if any(slc):
                self.viewer.show_event(ps[slc])
        return ps


########## end of file ##########
