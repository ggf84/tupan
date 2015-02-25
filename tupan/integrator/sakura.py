# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import logging
from .base import Base, power_of_two
from ..lib import extensions as ext
from ..lib.utils.timing import timings, bind_all


LOGGER = logging.getLogger(__name__)


@timings
def sakura_step(ps, dt, kernel=ext.Sakura()):
    """

    """
    ps.pos += ps.vel * dt / 2

    kernel(ps, ps, dt=dt/2, flag=-1)
    ps.pos += ps.dpos
    ps.vel += ps.dvel

    kernel(ps, ps, dt=dt/2, flag=+1)
    ps.pos += ps.dpos
    ps.vel += ps.dvel

    ps.pos += ps.vel * dt / 2

    return ps


@bind_all(timings)
class Sakura(Base):
    """

    """
    PROVIDED_METHODS = ['sakura', 'asakura', ]

    def __init__(self, eta, time, ps, method, **kwargs):
        """

        """
        super(Sakura, self).__init__(eta, time, ps, **kwargs)
        self.method = method
        self.e0 = None

    def initialize(self, t_end):
        """

        """
        ps = self.ps
        LOGGER.info("Initializing '%s' integrator at "
                    "t_curr = %g and t_end = %g.",
                    self.method, ps.t_curr, t_end)

        if self.reporter:
            self.reporter.diagnostic_report(ps)
        if self.dumpper:
            self.dumpper.init_worldline(ps)
        if self.viewer:
            self.viewer.show_event(ps)

        self.is_initialized = True

    def finalize(self, t_end):
        """

        """
        ps = self.ps
        LOGGER.info("Finalizing '%s' integrator at "
                    "t_curr = %g and t_end = %g.",
                    self.method, ps.t_curr, t_end)

        if self.viewer:
            self.viewer.show_event(ps)
            self.viewer.enter_main_loop()

    def get_sakura_tstep(self, ps, eta, dt):
        """

        """
        ps.set_tstep(ps, eta)

        iw2_a = (eta / ps.tstep)**2
        iw2_b = (eta / ps.tstepij)**2

        diw2 = (iw2_a - iw2_b)

        w2_sakura = diw2.max()
        dt_sakura = eta / (1 + w2_sakura)**0.5

        ps.tstep[...] = dt_sakura

        return power_of_two(ps, dt)

    def do_step(self, ps, dt):
        """

        """
#        p0 = p.copy()
#        if self.e0 is None:
#            self.e0 = p0.kinetic_energy + p0.potential_energy
#        de = [1]
#        tol = dt**2
#        nsteps = 1
#
#        while abs(de[0]) > tol:
#            p = p0.copy()
#            dt = dt / nsteps
#            for i in range(nsteps):
#                p = sakura_step(p, dt)
#                e1 = p.kinetic_energy + p.potential_energy
#                de[0] = e1/self.e0 - 1
#                if abs(de[0]) > tol:
# #                   nsteps += (nsteps+1)//2
#                    nsteps *= 2
# #                   print(nsteps, de, tol)
#                    break

        if 'asakura' in self.method:
            dt = self.get_sakura_tstep(ps, self.eta, dt)
        ps = sakura_step(ps, dt)

        ps.tstep[...] = dt
        ps.time += dt
        ps.nstep += 1
        type(ps).t_curr += dt
        self.dump(dt, ps)
        return ps


# -- End of File --
