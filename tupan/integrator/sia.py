#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import logging
import heapq
import math
import numpy as np
from .bios import sakura
from ..lib.utils.timing import decallmethods, timings


__all__ = ["SIA"]

logger = logging.getLogger(__name__)


#
# global constants
#
pn_order = 0
clight = None


#
# split
#
@timings
def split(isys, condition):
    """
    Splits the particle's system in slow/fast components.
    """
    slow = isys[condition]
    fast = isys[~condition]

    # prevents the occurrence of a slow level with only one particle.
    if slow.n == 1:
        fast.append(slow)
        slow = slow[:0]

    # prevents the occurrence of a fast level with only one particle.
    if fast.n == 1:
        slow.append(fast)
        fast = fast[:0]

    if slow.n + fast.n != isys.n:
        logger.error(
            "slow.n + fast.n != isys.n: %d, %d, %d.", slow.n, fast.n, isys.n)

    return slow, fast


#
# join
#
@timings
def join(slow, fast):
    """
    Joins the slow/fast components of a particle's system.
    """
    if slow.n == 0:
        return fast
    if fast.n == 0:
        return slow
    isys = slow[:]
    isys.append(fast)
    return isys


#
# drift_n
#
@timings
def drift_n(isys, tau):
    """
    Drift operator for Newtonian quantities.
    """
    if isys.n:
        for iobj in isys.members:
            iobj.x += tau * iobj.vx
            iobj.y += tau * iobj.vy
            iobj.z += tau * iobj.vz
    return isys


#
# kick_n
#
@timings
def kick_n(isys, jsys, tau):
    """
    Kick operator for Newtonian quantities.
    """
    if isys.n and jsys.n:
        for iobj in isys.members:
            (iax, iay, iaz) = iobj.get_acc(jsys)
            iobj.vx += tau * iax
            iobj.vy += tau * iay
            iobj.vz += tau * iaz
    return isys


#
# drift_pn
#
@timings
def drift_pn(isys, tau):
    """
    Drift operator for post-Newtonian quantities.
    """
    isys = drift_n(isys, tau)
    if isys.n:
        isys.evolve_rcom_pn_shift(tau)
    return isys


#
# kick_pn
#
@timings
def kick_pn(ip, jp, tau, pn_order, clight):
    """
    Kick operator for post-Newtonian quantities.
    """
    if ip.n and jp.n:
        ip = kick_n(ip, jp, tau / 2)

        ip.vx += ip.pn_dvx
        ip.vy += ip.pn_dvy
        ip.vz += ip.pn_dvz

        ip.evolve_ke_pn_shift(tau / 2)
        ip.evolve_lmom_pn_shift(tau / 2)
        ip.evolve_amom_pn_shift(tau / 2)

        (pnax, pnay, pnaz) = ip.get_pnacc(jp, pn_order, clight)
        g = 2 * tau * 0
        ip.pn_dvx = (tau * pnax - (1 - g) * ip.pn_dvx) / (1 + g)
        ip.pn_dvy = (tau * pnay - (1 - g) * ip.pn_dvy) / (1 + g)
        ip.pn_dvz = (tau * pnaz - (1 - g) * ip.pn_dvz) / (1 + g)

        ip.evolve_amom_pn_shift(tau / 2)
        ip.evolve_lmom_pn_shift(tau / 2)
        ip.evolve_ke_pn_shift(tau / 2)

        ip.vz += ip.pn_dvz
        ip.vy += ip.pn_dvy
        ip.vx += ip.pn_dvx

        ip = kick_n(ip, jp, tau / 2)
    return ip


#
# drift
#
@timings
def drift(isys, tau):
    """
    Drift operator.
    """
    if pn_order > 0:
        return drift_pn(isys, tau)
    return drift_n(isys, tau)


#
# kick
#
@timings
def kick(isys, jsys, tau, pn):
    """
    Kick operator.
    """
    if pn and pn_order > 0:
        return kick_pn(isys, jsys, tau, pn_order, clight)
    return kick_n(isys, jsys, tau)


#
# dkd21
#
dkd21_coefs = ([1.0],
               [0.5])


@timings
def base_dkd21(isys, tau):
    """
    Standard dkd21 operator.
    """
    k, d = dkd21_coefs

    isys = drift(isys, d[0] * tau)
    isys = kick(isys, isys, k[0] * tau, pn=True)
    isys = drift(isys, d[0] * tau)

    return isys


#
# dkd22
#
dkd22_coefs = ([0.5],
               [0.1931833275037836,
                0.6136333449924328])


@timings
def base_dkd22(isys, tau):
    """
    Standard dkd22 operator.
    """
    k, d = dkd22_coefs

    isys = drift(isys, d[0] * tau)
    isys = kick(isys, isys, k[0] * tau, pn=True)
    isys = drift(isys, d[1] * tau)
    isys = kick(isys, isys, k[0] * tau, pn=True)
    isys = drift(isys, d[0] * tau)

    return isys


#
# dkd43
#
dkd43_coefs = ([1.3512071919596575,
                -1.7024143839193150],
               [0.6756035959798288,
                -0.17560359597982877])


@timings
def base_dkd43(isys, tau):
    """
    Standard dkd43 operator.
    """
    k, d = dkd43_coefs

    isys = drift(isys, d[0] * tau)
    isys = kick(isys, isys, k[0] * tau, pn=True)
    isys = drift(isys, d[1] * tau)
    isys = kick(isys, isys, k[1] * tau, pn=True)
    isys = drift(isys, d[1] * tau)
    isys = kick(isys, isys, k[0] * tau, pn=True)
    isys = drift(isys, d[0] * tau)

    return isys


#
# kdk
#
@timings
def kdk(isys, tau):
    """
    Kick-Drift-Kick operator.
    """
    isys = kick(isys, isys, tau / 2, pn=True)
    isys = drift(isys, tau)
    isys = kick(isys, isys, tau / 2, pn=True)
    return isys


#
# kick_sf
#
@timings
def kick_sf(slow, fast, tau):
    """
    Slow<->Fast Kick operator.
    """
    slow = kick(slow, fast, tau, pn=True)
    fast = kick(fast, slow, tau, pn=True)
    return slow, fast


class Base(object):
    """
    A base class for the Symplectic Integration Algorithms (SIAs).
    """
    def __init__(self, eta, time, particles, **kwargs):
        self.eta = eta
        self.time = time
        self.particles = particles
        self.is_initialized = False

        global pn_order
        global clight
        pn_order = kwargs.pop("pn_order", 0)
        clight = kwargs.pop("clight", None)
        if pn_order > 0 and clight is None:
            raise TypeError("'clight' is not defined. Please set the speed of "
                            "light argument 'clight' when using 'pn_order' > 0.")

        self.reporter = kwargs.pop("reporter", None)
        self.viewer = kwargs.pop("viewer", None)
        self.dumpper = kwargs.pop("dumpper", None)
        self.dump_freq = kwargs.pop("dump_freq", 1)
        if kwargs:
            msg = "{0}.__init__ received unexpected keyword arguments: {1}."
            raise TypeError(msg.format(type(
                self).__name__, ", ".join(kwargs.keys())))


@decallmethods(timings)
class SIA(Base):
    """

    """
    PROVIDED_METHODS = ['sia.dkd21std', 'sia.dkd21shr', 'sia.dkd21hcc',
                        'sia.dkd22std', 'sia.dkd22shr', 'sia.dkd22hcc',
                        'sia.dkd43std', 'sia.dkd43shr', 'sia.dkd43hcc',
                        'sia.dkd44std', 'sia.dkd44shr', 'sia.dkd44hcc',
                        'sia.dkd45std', 'sia.dkd45shr', 'sia.dkd45hcc',
                        'sia.dkd46std', 'sia.dkd46shr', 'sia.dkd46hcc',
                        'sia.dkd67std', 'sia.dkd67shr', 'sia.dkd67hcc',
                        'sia.dkd69std', 'sia.dkd69shr', 'sia.dkd69hcc',
                        ]

    def __init__(self, eta, time, particles, method, **kwargs):
        super(SIA, self).__init__(eta, time, particles, **kwargs)
        self.method = method

    def get_base_tstep(self, t_end):
        self.tstep = self.eta
        if abs(self.time + self.tstep) > t_end:
            self.tstep = math.copysign(t_end - abs(self.time), self.eta)
        return self.tstep

    def initialize(self, t_end):
        logger.info("Initializing '%s' integrator.", self.method)

        p = self.particles

        if self.dumpper:
            self.dumpper.dump_worldline(p)

        self.is_initialized = True

    def finalize(self, t_end):
        logger.info("Finalizing '%s' integrator.", self.method)

        p = self.particles

        if self.reporter:
            self.reporter.report(self.time, p)

    def get_min_block_tstep(self, p, tau):
        min_tstep = p.min_tstep()

        power = int(np.log2(min_tstep) - 1)
        min_block_tstep = 2.0**power

        next_time = self.time + min_block_tstep
        if next_time % min_block_tstep != 0:
            min_block_tstep /= 2

        self.tstep = min_block_tstep
        if self.tstep > abs(tau):
            self.tstep = tau

        self.tstep = math.copysign(self.tstep, self.eta)
        return self.tstep

    ### evolve_step
    def evolve_step(self, t_end):
        """

        """
        if not self.is_initialized:
            self.initialize(t_end)

        p = self.particles
        tau = self.get_base_tstep(t_end)

        if self.reporter:
            self.reporter.report(self.time, p)
        self.wl = p[:0]

        if self.method == "sia.dkd21std":
            p = self.dkd21(p, tau, False, True)
        elif self.method == "sia.dkd21shr":
            p = self.dkd21(p, tau, True, True)
        elif self.method == "sia.dkd21hcc":
            p = self.dkd21(p, tau, True, False)
        elif self.method == "sia.dkd22std":
            p = self.dkd22(p, tau, False, True)
        elif self.method == "sia.dkd22shr":
            p = self.dkd22(p, tau, True, True)
        elif self.method == "sia.dkd22hcc":
            p = self.dkd22(p, tau, True, False)
        elif self.method == "sia.dkd43std":
            p = self.dkd43(p, tau, False, True)
        elif self.method == "sia.dkd43shr":
            p = self.dkd43(p, tau, True, True)
        elif self.method == "sia.dkd43hcc":
            p = self.dkd43(p, tau, True, False)
        elif self.method == "sia.dkd44std":
            p = self.dkd44(p, tau, False, True)
        elif self.method == "sia.dkd44shr":
            p = self.dkd44(p, tau, True, True)
        elif self.method == "sia.dkd44hcc":
            p = self.dkd44(p, tau, True, False)
        elif self.method == "sia.dkd45std":
            p = self.dkd45(p, tau, False, True)
        elif self.method == "sia.dkd45shr":
            p = self.dkd45(p, tau, True, True)
        elif self.method == "sia.dkd45hcc":
            p = self.dkd45(p, tau, True, False)
        elif self.method == "sia.dkd46std":
            p = self.dkd46(p, tau, False, True)
        elif self.method == "sia.dkd46shr":
            p = self.dkd46(p, tau, True, True)
        elif self.method == "sia.dkd46hcc":
            p = self.dkd46(p, tau, True, False)
        elif self.method == "sia.dkd67std":
            p = self.dkd67(p, tau, False, True)
        elif self.method == "sia.dkd67shr":
            p = self.dkd67(p, tau, True, True)
        elif self.method == "sia.dkd67hcc":
            p = self.dkd67(p, tau, True, False)
        elif self.method == "sia.dkd69std":
            p = self.dkd69(p, tau, False, True)
        elif self.method == "sia.dkd69shr":
            p = self.dkd69(p, tau, True, True)
        elif self.method == "sia.dkd69hcc":
            p = self.dkd69(p, tau, True, False)
        elif self.method == 0:
            pass
##            p = self.heapq0(p, tau, True)
        else:
            raise ValueError("Unexpected method: {0}".format(self.method))

        if self.dumpper:
            self.dumpper.dump_worldline(self.wl)

        self.time += self.tstep
        self.particles = p

    #
    # dkd21[std,shr,hcc] method -- D.K.D
    #
    def dkd21__(self, p, tau, update_tstep, shared_tstep=False):
        if p.n:

            if update_tstep:
                p.update_tstep(p, self.eta)
                if shared_tstep:
                    tau = self.get_min_block_tstep(p, tau)
            flag = 0 if shared_tstep and not update_tstep else 1

            if p.n == 2:
                flag = 0

            slow, fast = split(p, abs(p.tstep) >= abs(tau*flag))

            k, d = dkd21_coefs
            #
            if fast.n:
                fast = self.dkd21(fast, d[0] * tau, True)
                if slow.n:
                    slow = sakura(slow, d[0] * tau)
                fast, slow = kick_sf(fast, slow, k[0] * tau)
                if slow.n:
                    slow = sakura(slow, d[0] * tau)
                fast = self.dkd21(fast, d[0] * tau, True)
            else:
                if slow.n:
                    slow = sakura(slow, tau)
            #

            if slow.n:
                slow.tstep = tau
                slow.time += tau
                slow.nstep += 1
                wp = slow[slow.time % (self.dump_freq * tau) == 0]
                if wp.n:
                    self.wl.append(wp[:])

            p = join(slow, fast)

        return p

    #
    # dkd21[std,shr,hcc] method -- D.K.D
    #
    def dkd21(self, p, tau, update_tstep, shared_tstep=False):
        if p.n:

            if update_tstep:
                p.update_tstep(p, self.eta)
                if shared_tstep:
                    tau = self.get_min_block_tstep(p, tau)
            flag = 0 if shared_tstep and not update_tstep else 1

            slow, fast = split(p, abs(p.tstep) >= abs(tau*flag))

            k, d = dkd21_coefs
            #
            if fast.n:
                fast = self.dkd21(fast, d[0] * tau / 2, True)
                slow = base_dkd21(slow, d[0] * tau)
                fast = self.dkd21(fast, d[0] * tau / 2, True)
                slow, fast = kick_sf(slow, fast, k[0] * tau)
                fast = self.dkd21(fast, d[0] * tau / 2, True)
                slow = base_dkd21(slow, d[0] * tau)
                fast = self.dkd21(fast, d[0] * tau / 2, True)

            else:
                slow = base_dkd21(slow, tau)
            #

            if slow.n:
                slow.tstep = tau
                slow.time += tau
                slow.nstep += 1
                wp = slow[slow.time % (self.dump_freq * tau) == 0]
                if wp.n:
                    self.wl.append(wp[:])

            p = join(slow, fast)

        return p

    #
    # dkd22[std,shr,hcc] method -- D.K.D.K.D
    #
    def dkd22(self, p, tau, update_tstep, shared_tstep=False):
        if p.n:

            if update_tstep:
                p.update_tstep(p, self.eta)
                if shared_tstep:
                    tau = self.get_min_block_tstep(p, tau)
            flag = 0 if shared_tstep and not update_tstep else 1

            slow, fast = split(p, abs(p.tstep) >= abs(tau*flag))

            k, d = dkd22_coefs
            #
            if fast.n:
                fast, slow = kick_sf(fast, slow, d[0] * tau)

                fast = self.dkd22(fast, k[0] * tau / 2, True)
                slow = base_dkd22(slow, k[0] * tau)
                fast = self.dkd22(fast, k[0] * tau / 2, True)

                slow, fast = kick_sf(slow, fast, d[1] * tau)

                fast = self.dkd22(fast, k[0] * tau / 2, True)
                slow = base_dkd22(slow, k[0] * tau)
                fast = self.dkd22(fast, k[0] * tau / 2, True)

                fast, slow = kick_sf(fast, slow, d[0] * tau)
            else:
                slow = base_dkd22(slow, tau)
            #

            if slow.n:
                slow.tstep = tau
                slow.time += tau
                slow.nstep += 1
                wp = slow[slow.time % (self.dump_freq * tau) == 0]
                if wp.n:
                    self.wl.append(wp[:])

            p = join(slow, fast)

        return p

    #
    # dkd43[std,shr,hcc] method -- D.K.D.K.D.K.D
    #
    def dkd43(self, p, tau, update_tstep, shared_tstep=False):
        if p.n:

            if update_tstep:
                p.update_tstep(p, self.eta)
                if shared_tstep:
                    tau = self.get_min_block_tstep(p, tau)
            flag = 0 if shared_tstep and not update_tstep else 1

            slow, fast = split(p, abs(p.tstep) >= abs(tau*flag))

            k, d = dkd43_coefs
            #
            slow = drift(slow, d[0] * tau)

            fast = self.dkd43(fast, d[0] * tau / 3, True)
            fast = self.dkd43(fast, d[0] * tau / 3, True)
            fast = self.dkd43(fast, d[0] * tau / 3, True)
            fast, slow = kick_sf(fast, slow, k[0] * tau / 2)

            slow = kick(slow, slow, k[0] * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k[0] * tau / 2)
            fast = self.dkd43(fast, d[1] * tau / 2, True)

            slow = drift(slow, d[1] * tau)

            fast = self.dkd43(fast, d[1] * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k[1] * tau / 2)

            slow = kick(slow, slow, k[1] * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k[1] * tau / 2)
            fast = self.dkd43(fast, d[1] * tau / 2, True)

            slow = drift(slow, d[1] * tau)

            fast = self.dkd43(fast, d[1] * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k[0] * tau / 2)

            slow = kick(slow, slow, k[0] * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k[0] * tau / 2)
            fast = self.dkd43(fast, d[0] * tau / 3, True)
            fast = self.dkd43(fast, d[0] * tau / 3, True)
            fast = self.dkd43(fast, d[0] * tau / 3, True)

            slow = drift(slow, d[0] * tau)
            #

#            k, d = dkd43_coefs
#            #
#            if fast.n:
#                fast, slow = kick_sf(fast, slow, d[0] * tau)
#
#                fast = self.dkd43(fast, k[0] * tau / 2, True)
#                slow = base_dkd43(slow, k[0] * tau)
#                fast = self.dkd43(fast, k[0] * tau / 2, True)
#
#                slow, fast = kick_sf(slow, fast, d[1] * tau)
#
#                fast = self.dkd43(fast, k[1] * tau / 2, True)
#                slow = base_dkd43(slow, k[1] * tau)
#                fast = self.dkd43(fast, k[1] * tau / 2, True)
#
#                fast, slow = kick_sf(fast, slow, d[1] * tau)
#
#                fast = self.dkd43(fast, k[0] * tau / 2, True)
#                slow = base_dkd43(slow, k[0] * tau)
#                fast = self.dkd43(fast, k[0] * tau / 2, True)
#
#                slow, fast = kick_sf(slow, fast, d[0] * tau)
#            else:
#                slow = base_dkd43(slow, tau)
#            #

            if slow.n:
                slow.tstep = tau
                slow.time += tau
                slow.nstep += 1
                wp = slow[slow.time % (self.dump_freq * tau) == 0]
                if wp.n:
                    self.wl.append(wp[:])

            p = join(slow, fast)

        return p

    #
    # dkd44[std,shr,hcc] method -- D.K.D.K.D.K.D.K.D
    #
    def dkd44(self, p, tau, update_tstep,
              shared_tstep=False,
              k0=0.7123418310626056,
              k1=-0.21234183106260562,
              d0=0.1786178958448091,
              d1=-0.06626458266981843,
              d2=0.7752933736500186):
        if p.n:

            if update_tstep:
                p.update_tstep(p, self.eta)
                if shared_tstep:
                    tau = self.get_min_block_tstep(p, tau)
            flag = 0 if shared_tstep and not update_tstep else 1

            slow, fast = split(p, abs(p.tstep) >= abs(tau*flag))

            #
            slow = drift(slow, d0 * tau)

            fast = self.dkd44(fast, d0 * tau, True)
            fast, slow = kick_sf(fast, slow, k0 * tau / 2)

            slow = kick(slow, slow, k0 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k0 * tau / 2)
            fast = self.dkd44(fast, d1 * tau / 2, True)

            slow = drift(slow, d1 * tau)

            fast = self.dkd44(fast, d1 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k1 * tau / 2)

            slow = kick(slow, slow, k1 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k1 * tau / 2)
            fast = self.dkd44(fast, d2 * tau / 4, True)
            fast = self.dkd44(fast, d2 * tau / 4, True)

            slow = drift(slow, d2 * tau)

            fast = self.dkd44(fast, d2 * tau / 4, True)
            fast = self.dkd44(fast, d2 * tau / 4, True)
            fast, slow = kick_sf(fast, slow, k1 * tau / 2)

            slow = kick(slow, slow, k1 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k1 * tau / 2)
            fast = self.dkd44(fast, d1 * tau / 2, True)

            slow = drift(slow, d1 * tau)

            fast = self.dkd44(fast, d1 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k0 * tau / 2)

            slow = kick(slow, slow, k0 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k0 * tau / 2)
            fast = self.dkd44(fast, d0 * tau, True)

            slow = drift(slow, d0 * tau)
            #

            if slow.n:
                slow.tstep = tau
                slow.time += tau
                slow.nstep += 1
                wp = slow[slow.time % (self.dump_freq * tau) == 0]
                if wp.n:
                    self.wl.append(wp[:])

            p = join(slow, fast)

        return p

    #
    # dkd45[std,shr,hcc] method -- D.K.D.K.D.K.D.K.D.K.D
    #
    def dkd45(self, p, tau, update_tstep,
              shared_tstep=False,
              k0=-0.0844296195070715,
              k1=0.354900057157426,
              k2=0.459059124699291,
              d0=0.2750081212332419,
              d1=-0.1347950099106792,
              d2=0.35978688867743724):
        if p.n:

            if update_tstep:
                p.update_tstep(p, self.eta)
                if shared_tstep:
                    tau = self.get_min_block_tstep(p, tau)
            flag = 0 if shared_tstep and not update_tstep else 1

            slow, fast = split(p, abs(p.tstep) >= abs(tau*flag))

            #
            slow = drift(slow, d0 * tau)

            fast = self.dkd45(fast, d0 * tau, True)
            fast, slow = kick_sf(fast, slow, k0 * tau / 2)

            slow = kick(slow, slow, k0 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k0 * tau / 2)
            fast = self.dkd45(fast, d1 * tau / 2, True)

            slow = drift(slow, d1 * tau)

            fast = self.dkd45(fast, d1 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k1 * tau / 2)

            slow = kick(slow, slow, k1 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k1 * tau / 2)
            fast = self.dkd45(fast, d2 * tau / 2, True)

            slow = drift(slow, d2 * tau)

            fast = self.dkd45(fast, d2 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k2 * tau / 2)

            slow = kick(slow, slow, k2 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k2 * tau / 2)
            fast = self.dkd45(fast, d2 * tau / 2, True)

            slow = drift(slow, d2 * tau)

            fast = self.dkd45(fast, d2 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k1 * tau / 2)

            slow = kick(slow, slow, k1 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k1 * tau / 2)
            fast = self.dkd45(fast, d1 * tau / 2, True)

            slow = drift(slow, d1 * tau)

            fast = self.dkd45(fast, d1 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k0 * tau / 2)

            slow = kick(slow, slow, k0 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k0 * tau / 2)
            fast = self.dkd45(fast, d0 * tau, True)

            slow = drift(slow, d0 * tau)
            #

            if slow.n:
                slow.tstep = tau
                slow.time += tau
                slow.nstep += 1
                wp = slow[slow.time % (self.dump_freq * tau) == 0]
                if wp.n:
                    self.wl.append(wp[:])

            p = join(slow, fast)

        return p

    #
    # dkd46[std,shr,hcc] method -- D.K.D.K.D.K.D.K.D.K.D.K.D
    #
    def dkd46(self, p, tau, update_tstep,
              shared_tstep=False,
              k0=0.209515106613362,
              k1=-0.143851773179818,
              k2=0.434336666566456,
              d0=0.0792036964311957,
              d1=0.353172906049774,
              d2=-0.0420650803577195,
              d3=0.21937695575349958):
        if p.n:

            if update_tstep:
                p.update_tstep(p, self.eta)
                if shared_tstep:
                    tau = self.get_min_block_tstep(p, tau)
            flag = 0 if shared_tstep and not update_tstep else 1

            slow, fast = split(p, abs(p.tstep) >= abs(tau*flag))

            #
            slow = drift(slow, d0 * tau)

            fast = self.dkd46(fast, d0 * tau, True)
            fast, slow = kick_sf(fast, slow, k0 * tau / 2)

            slow = kick(slow, slow, k0 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k0 * tau / 2)
            fast = self.dkd46(fast, d1 * tau / 2, True)

            slow = drift(slow, d1 * tau)

            fast = self.dkd46(fast, d1 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k1 * tau / 2)

            slow = kick(slow, slow, k1 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k1 * tau / 2)
            fast = self.dkd46(fast, d2 * tau / 2, True)

            slow = drift(slow, d2 * tau)

            fast = self.dkd46(fast, d2 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k2 * tau / 2)

            slow = kick(slow, slow, k2 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k2 * tau / 2)
            fast = self.dkd46(fast, d3 * tau / 2, True)

            slow = drift(slow, d3 * tau)

            fast = self.dkd46(fast, d3 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k2 * tau / 2)

            slow = kick(slow, slow, k2 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k2 * tau / 2)
            fast = self.dkd46(fast, d2 * tau / 2, True)

            slow = drift(slow, d2 * tau)

            fast = self.dkd46(fast, d2 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k1 * tau / 2)

            slow = kick(slow, slow, k1 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k1 * tau / 2)
            fast = self.dkd46(fast, d1 * tau / 2, True)

            slow = drift(slow, d1 * tau)

            fast = self.dkd46(fast, d1 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k0 * tau / 2)

            slow = kick(slow, slow, k0 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k0 * tau / 2)
            fast = self.dkd46(fast, d0 * tau, True)

            slow = drift(slow, d0 * tau)
            #

            if slow.n:
                slow.tstep = tau
                slow.time += tau
                slow.nstep += 1
                wp = slow[slow.time % (self.dump_freq * tau) == 0]
                if wp.n:
                    self.wl.append(wp[:])

            p = join(slow, fast)

        return p

    #
    # dkd67[std,shr,hcc] method -- D.K.D.K.D.K.D.K.D.K.D.K.D.K.D
    #
    def dkd67(self, p, tau, update_tstep,
              shared_tstep=False,
              k0=0.7845136104775573,
              k1=0.23557321335935813,
              k2=-1.177679984178871,
              k3=1.3151863206839112,
              d0=0.39225680523877865,
              d1=0.5100434119184577,
              d2=-0.47105338540975644,
              d3=0.06875316825252015):
        if p.n:

            if update_tstep:
                p.update_tstep(p, self.eta)
                if shared_tstep:
                    tau = self.get_min_block_tstep(p, tau)
            flag = 0 if shared_tstep and not update_tstep else 1

            slow, fast = split(p, abs(p.tstep) >= abs(tau*flag))

            #
            slow = drift(slow, d0 * tau)

            fast = self.dkd67(fast, d0 * tau / 2, True)
            fast = self.dkd67(fast, d0 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k0 * tau / 2)

            slow = kick(slow, slow, k0 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k0 * tau / 2)
            fast = self.dkd67(fast, d1 * tau / 2, True)

            slow = drift(slow, d1 * tau)

            fast = self.dkd67(fast, d1 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k1 * tau / 2)

            slow = kick(slow, slow, k1 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k1 * tau / 2)
            fast = self.dkd67(fast, d2 * tau / 2, True)

            slow = drift(slow, d2 * tau)

            fast = self.dkd67(fast, d2 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k2 * tau / 2)

            slow = kick(slow, slow, k2 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k2 * tau / 2)
            fast = self.dkd67(fast, d3 * tau / 2, True)

            slow = drift(slow, d3 * tau)

            fast = self.dkd67(fast, d3 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k3 * tau / 2)

            slow = kick(slow, slow, k3 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k3 * tau / 2)
            fast = self.dkd67(fast, d3 * tau / 2, True)

            slow = drift(slow, d3 * tau)

            fast = self.dkd67(fast, d3 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k2 * tau / 2)

            slow = kick(slow, slow, k2 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k2 * tau / 2)
            fast = self.dkd67(fast, d2 * tau / 2, True)

            slow = drift(slow, d2 * tau)

            fast = self.dkd67(fast, d2 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k1 * tau / 2)

            slow = kick(slow, slow, k1 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k1 * tau / 2)
            fast = self.dkd67(fast, d1 * tau / 2, True)

            slow = drift(slow, d1 * tau)

            fast = self.dkd67(fast, d1 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k0 * tau / 2)

            slow = kick(slow, slow, k0 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k0 * tau / 2)
            fast = self.dkd67(fast, d0 * tau / 2, True)
            fast = self.dkd67(fast, d0 * tau / 2, True)

            slow = drift(slow, d0 * tau)
            #

            if slow.n:
                slow.tstep = tau
                slow.time += tau
                slow.nstep += 1
                wp = slow[slow.time % (self.dump_freq * tau) == 0]
                if wp.n:
                    self.wl.append(wp[:])

            p = join(slow, fast)

        return p

    #
    # dkd69[std,shr,hcc] method -- D.K.D.K.D.K.D.K.D.K.D.K.D.K.D.K.D.K.D
    #
    def dkd69(self, p, tau, update_tstep,
              shared_tstep=False,
              k0=0.39103020330868477,
              k1=0.334037289611136,
              k2=-0.7062272811875614,
              k3=0.08187754964805945,
              k4=0.7985644772393624,
              d0=0.19551510165434238,
              d1=0.3625337464599104,
              d2=-0.1860949957882127,
              d3=-0.31217486576975095,
              d4=0.44022101344371095):
        if p.n:

            if update_tstep:
                p.update_tstep(p, self.eta)
                if shared_tstep:
                    tau = self.get_min_block_tstep(p, tau)
            flag = 0 if shared_tstep and not update_tstep else 1

            slow, fast = split(p, abs(p.tstep) >= abs(tau*flag))

            #
            slow = drift(slow, d0 * tau)

            fast = self.dkd69(fast, d0 * tau, True)
            fast, slow = kick_sf(fast, slow, k0 * tau / 2)

            slow = kick(slow, slow, k0 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k0 * tau / 2)
            fast = self.dkd69(fast, d1 * tau / 2, True)

            slow = drift(slow, d1 * tau)

            fast = self.dkd69(fast, d1 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k1 * tau / 2)

            slow = kick(slow, slow, k1 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k1 * tau / 2)
            fast = self.dkd69(fast, d2 * tau / 2, True)

            slow = drift(slow, d2 * tau)

            fast = self.dkd69(fast, d2 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k2 * tau / 2)

            slow = kick(slow, slow, k2 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k2 * tau / 2)
            fast = self.dkd69(fast, d3 * tau / 2, True)

            slow = drift(slow, d3 * tau)

            fast = self.dkd69(fast, d3 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k3 * tau / 2)

            slow = kick(slow, slow, k3 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k3 * tau / 2)
            fast = self.dkd69(fast, d4 * tau / 2, True)

            slow = drift(slow, d4 * tau)

            fast = self.dkd69(fast, d4 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k4 * tau / 2)

            slow = kick(slow, slow, k4 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k4 * tau / 2)
            fast = self.dkd69(fast, d4 * tau / 2, True)

            slow = drift(slow, d4 * tau)

            fast = self.dkd69(fast, d4 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k3 * tau / 2)

            slow = kick(slow, slow, k3 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k3 * tau / 2)
            fast = self.dkd69(fast, d3 * tau / 2, True)

            slow = drift(slow, d3 * tau)

            fast = self.dkd69(fast, d3 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k2 * tau / 2)

            slow = kick(slow, slow, k2 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k2 * tau / 2)
            fast = self.dkd69(fast, d2 * tau / 2, True)

            slow = drift(slow, d2 * tau)

            fast = self.dkd69(fast, d2 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k1 * tau / 2)

            slow = kick(slow, slow, k1 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k1 * tau / 2)
            fast = self.dkd69(fast, d1 * tau / 2, True)

            slow = drift(slow, d1 * tau)

            fast = self.dkd69(fast, d1 * tau / 2, True)
            fast, slow = kick_sf(fast, slow, k0 * tau / 2)

            slow = kick(slow, slow, k0 * tau, pn=True)

            slow, fast = kick_sf(slow, fast, k0 * tau / 2)
            fast = self.dkd69(fast, d0 * tau, True)

            slow = drift(slow, d0 * tau)
            #

            if slow.n:
                slow.tstep = tau
                slow.time += tau
                slow.nstep += 1
                wp = slow[slow.time % (self.dump_freq * tau) == 0]
                if wp.n:
                    self.wl.append(wp[:])

            p = join(slow, fast)

        return p

    #
    # the following methods are for experimental purposes only.
    #
    ### heapq0
    def heapq0(self, p, tau, update_tstep):

        dt = tau
        t = self.time
        ph = []
        while p.n > 0:
            slow, fast, indexing = split(dt, p)
            ph.append((t+dt, dt, p, slow, fast, indexing))
            dt /= 2
            p = fast
        heapq.heapify(ph)

        tstop = self.time+tau
        while ph[0][0] < tstop:
            (t, dt, p, slow, fast, indexing) = heapq.heappop(ph)
            if fast.n > 0:
                self.sf_dkd(slow, fast, dt)
                t += dt
                self.time = t

            self.merge(p, slow, fast, indexing)
            if update_tstep:
                p.update_tstep(p, self.eta)      # True/False
            if update_tstep:
                p.update_phi(p)      # True/False
            slow, fast, indexing = split(dt/2, p)

            if fast.n > 0:
                heapq.heappush(ph, (t+dt/2, dt/2, p, slow, fast, indexing))

            print(len(ph), p.n, p['body'].t_curr[0], self.time, t, dt)

        return p


#        tstop = tau
#        while pheap[0][0] < tstop:
#            tnext, tau, p, slow, fast, indexing = heapq.heappop(pheap)
#
#            if fast.n == 0: self.time += tau / 2
#            if slow.n > 0: self.sf_dkd(slow, fast, tau)
#            if fast.n == 0: self.time += tau / 2
#
#            self.merge(p, slow, fast, indexing)
#
#            if update_tstep: p.update_tstep(p, self.eta)      # True/False
#
#            slow, fast, indexing = split(tau/2, p)
#
#            if fast.n > 0:
#                tnext += tau/2
#                heapq.heappush(p, (tnext, tau/2, p, slow, fast, indexing))
#            else:
#                tnext += tau
#                heapq.heappush(p, (tnext, tau, p, slow, fast, indexing))



########## end of file ##########
