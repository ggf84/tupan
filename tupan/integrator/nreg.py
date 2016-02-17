# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import logging
from .base import Base
from ..lib import extensions as ext
from ..lib.utils.timing import timings, bind_all


LOGGER = logging.getLogger(__name__)


@timings
def nreg_x(ps, dt, kernel=ext.get_kernel('NregX')):
    """

    """
    mtot = ps.total_mass
    kernel(ps, ps, dt=dt)
    ps.pos[...] = ps.mr / mtot
    pe = 0.5 * ps.u.sum()
    type(ps).U = pe
    return ps, dt

#    ps.pos += dt * ps.vel
#    type(ps).U = -ps.potential_energy
#    ps.set_acc(ps)
#    return ps, dt


@timings
def nreg_v(ps, dt, kernel=ext.get_kernel('NregV')):
    """

    """
#    type(ps).W += 0.5 * dt * (ps.mass * ps.vel * ps.acc).sum()
    mtot = ps.total_mass
    kernel(ps, ps, dt=dt)
    ps.vel[...] = ps.mv / mtot
    ke = 0.25 * ps.mk.sum() / mtot
    type(ps).W = (ke - ps.E0)
#    type(ps).W += 0.5 * dt * (ps.mass * ps.vel * ps.acc).sum()
    return ps

# #    type(ps).W += 0.5 * dt * (ps.mass * ps.vel * ps.acc).sum()
#     ps.vel += dt * ps.acc
#     type(ps).W = (ps.kinetic_energy - ps.E0)
# #    type(ps).W += 0.5 * dt * (ps.mass * ps.vel * ps.acc).sum()
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

    def __init__(self, ps, eta, dt_max, t_begin, method, **kwargs):
        """

        """
        super(NREG, self).__init__(ps, eta, dt_max,
                                   t_begin, method, **kwargs)

        if 'anreg' in self.method:
            self.update_tstep = True
            self.shared_tstep = True
        else:
            self.update_tstep = False
            self.shared_tstep = True

        type(ps).E0 = ps.kinetic_energy + ps.potential_energy
        type(ps).W = -ps.potential_energy
        type(ps).U = -ps.potential_energy
        type(ps).S = -ps.potential_energy

    def do_step(self, ps, h):
        """

        """
        if self.update_tstep:
            ps, dt = anreg_step(ps, h / 2)
        else:
            ps, dt = nreg_step(ps, h)

        type(ps).t_curr += dt
        ps.time += dt
        ps.nstep += 1
        return ps


# -- End of File --
