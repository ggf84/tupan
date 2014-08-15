# -*- coding: utf-8 -*-
#

"""
TODO.
"""


import logging
from .base import Base
from ..lib import extensions as ext
from ..lib.utils.timing import timings, bind_all


__all__ = ['NREG']


LOGGER = logging.getLogger(__name__)


@timings
def nreg_x(ps, dt, kernel=ext.NregX()):
    """

    """
    mtot = ps.total_mass
    kernel(ps, ps, dt=dt)
    ps.rx[...] = ps.mrx / mtot
    ps.ry[...] = ps.mry / mtot
    ps.rz[...] = ps.mrz / mtot
    pe = 0.5 * ps.u.sum()
    type(ps).U = pe
    return ps, dt

#    ps.rx += dt * ps.vx
#    ps.ry += dt * ps.vy
#    ps.rz += dt * ps.vz
#    type(ps).U = -ps.potential_energy
#    ps.set_acc(ps)
#    return ps, dt


@timings
def nreg_v(ps, dt, kernel=ext.NregV()):
    """

    """
#    type(ps).W += 0.5 * dt * (ps.mass * (ps.vx * ps.ax
#                                         + ps.vy * ps.ay
#                                         + ps.vz * ps.az)).sum()
    mtot = ps.total_mass
    kernel(ps, ps, dt=dt)
    ps.vx[...] = ps.mvx / mtot
    ps.vy[...] = ps.mvy / mtot
    ps.vz[...] = ps.mvz / mtot
    ke = 0.25 * ps.mk.sum() / mtot
    type(ps).W = (ke - ps.E0)
#    type(ps).W += 0.5 * dt * (ps.mass * (ps.vx * ps.ax
#                                         + ps.vy * ps.ay
#                                         + ps.vz * ps.az)).sum()
    return ps

# #    type(ps).W += 0.5 * dt * (ps.mass * (ps.vx * ps.ax
# #                                         + ps.vy * ps.ay
# #                                         + ps.vz * ps.az)).sum()
#     ps.vx += dt * ps.ax
#     ps.vy += dt * ps.ay
#     ps.vz += dt * ps.az
#     type(ps).W = (ps.kinetic_energy - ps.E0)
# #    type(ps).W += 0.5 * dt * (ps.mass * (ps.vx * ps.ax
# #                                         + ps.vy * ps.ay
# #                                         + ps.vz * ps.az)).sum()
#     return ps


@timings
def anreg_step(ps, h):
    """

    """
    ps, dt0 = nreg_x(ps, 0.5 * (h / ps.W))
    ps = nreg_v(ps, (h / ps.U))
    ps, dt1 = nreg_x(ps, 0.5 * (h / ps.W))

    return ps, (dt0 + dt1)


@timings
def nreg_step(ps, h):
    """

    """
    ps, dt0 = anreg_step(ps, 0.5 * (h * ps.S))
    type(ps).S = 1 / (2 / ps.W - 1 / ps.S)
    ps, dt1 = anreg_step(ps, 0.5 * (h * ps.S))

    return ps, (dt0 + dt1)


@bind_all(timings)
class NREG(Base):
    """

    """
    PROVIDED_METHODS = ['nreg', 'anreg', ]

    def __init__(self, eta, time, ps, method, **kwargs):
        """

        """
        super(NREG, self).__init__(eta, time, ps, **kwargs)
        self.method = method

    def initialize(self, t_end):
        """

        """
        ps = self.ps
        LOGGER.info("Initializing '%s' integrator at "
                    "t_curr = %g and t_end = %g.",
                    self.method, ps.t_curr, t_end)

        type(ps).E0 = ps.kinetic_energy + ps.potential_energy
        type(ps).W = -ps.potential_energy
        type(ps).U = -ps.potential_energy
        type(ps).S = -ps.potential_energy

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
        ps = self.ps
        LOGGER.info("Finalizing '%s' integrator at "
                    "t_curr = %g and t_end = %g.",
                    self.method, ps.t_curr, t_end)

        if self.viewer:
            self.viewer.show_event(ps)
            self.viewer.enter_main_loop()

    def do_step(self, ps, h):
        """

        """
        if 'anreg' in self.method:
            ps, dt = anreg_step(ps, h / 2)
        else:
            ps, dt = nreg_step(ps, h)

        ps.tstep[...] = dt
        ps.time += dt
        ps.nstep += 1
        type(ps).t_curr += dt
        self.dump(dt, ps)
        return ps


# -- End of File --
