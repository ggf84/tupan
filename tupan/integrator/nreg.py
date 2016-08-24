# -*- coding: utf-8 -*-
#

"""This module implements the N-body formulation of the algorithmic
regularization procedure by Mikkola & Tanikawa, MNRAS, 310, 745-749 (1999).
"""

from .base import Base


def drift(ps, h):
    """

    """
    W = ps.kinetic_energy + ps.B
    dt = h / W
    ps.time += dt
    ps.rdot[0] += ps.rdot[1] * dt
    return ps, W


def kick(ps, h):
    """

    """
    ps.set_acc(ps)
    U = -ps.potential_energy
    dt = h / U
    ps.rdot[1] += ps.rdot[2] * dt
    return ps, U


def a_nreg_step(ps, h):
    """

    """
    ps, _ = drift(ps, h/2)
    ps, _ = kick(ps, h)
    ps, _ = drift(ps, h/2)

    ps.nstep += 1
    return ps


def c_nreg_step(ps, h):
    """

    """
    try:
        U = ps.U
    except:
        U = -ps.potential_energy
        type(ps).U = U

    ps, _ = drift(ps, U * h/2)
    ps, _ = kick(ps, U * h)
    ps, W = drift(ps, U * h/2)
    type(ps).U = U = (W**2) / ps.U

    ps.nstep += 1
    return ps


class NREG(Base):
    """

    """
    PROVIDED_METHODS = [
        'c.nreg', 'a.nreg',
    ]

    def __init__(self, ps, eta, dt_max, t_begin, method, **kwargs):
        """

        """
        super(NREG, self).__init__(ps, eta, dt_max,
                                   t_begin, method, **kwargs)

        self.update_tstep = False
        self.shared_tstep = True

        T = ps.kinetic_energy
        U = -ps.potential_energy
        type(ps).B = U - T

    def do_step(self, ps, h):
        """

        """
        if 'c.' in self.method:
            return c_nreg_step(ps, h)
        elif 'a.' in self.method:
            return a_nreg_step(ps, h)
        else:
            raise NotImplemented(self.method)


# -- End of File --
