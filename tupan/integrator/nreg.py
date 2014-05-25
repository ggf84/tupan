# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function, division
import logging
from ..integrator import Base
from ..lib import extensions as ext
from ..lib.utils.timing import timings, bind_all


__all__ = ["NREG"]


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
    type(ps).t_curr += dt
    return ps

#    ps.rx += dt * ps.vx
#    ps.ry += dt * ps.vy
#    ps.rz += dt * ps.vz
#    type(ps).U = -ps.potential_energy
#    type(ps).t_curr += dt
#    ps.set_acc(ps)
#    return ps


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
    ps = nreg_x(ps, 0.5 * (h / ps.W))
    ps = nreg_v(ps, (h / ps.U))
    ps = nreg_x(ps, 0.5 * (h / ps.W))

    return ps


@timings
def nreg_step(ps, h):
    """

    """
    ps = anreg_step(ps, 0.5 * (h * ps.S))
    type(ps).S = 1/(2/ps.W - 1/ps.S)
    ps = anreg_step(ps, 0.5 * (h * ps.S))

    return ps


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

    def do_step(self, ps, dt):
        """

        """
        if "anreg" in self.method:
            t0 = ps.t_curr
            ps = anreg_step(ps, dt/2)
            t1 = ps.t_curr
        else:
            t0 = ps.t_curr
            ps = nreg_step(ps, dt)
            t1 = ps.t_curr

        ps.tstep[...] = t1 - t0
        ps.time += dt
        ps.nstep += 1
        if self.dumpper:
            slc = ps.time % (self.dump_freq * dt) == 0
            if any(slc):
                self.wl.append(ps[slc])
        if self.viewer:
            slc = ps.time % (self.gl_freq * dt) == 0
            if any(slc):
                self.viewer.show_event(ps[slc])
        return ps


# -- End of File --
