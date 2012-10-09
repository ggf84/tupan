#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import logging
import heapq
import math
import numpy as np
from ..lib.utils.timing import decallmethods, timings


__all__ = ["SIA"]

logger = logging.getLogger(__name__)


class Base(object):
    """
    A base class for the Symplectic Integration Algorithms (SIAs).
    """
    def __init__(self, eta, time, particles, **kwargs):
        self.eta = eta
        self.time = time
        self.particles = particles
        self.is_initialized = False

        self.pn_order = kwargs.pop("pn_order", 0)
        self.clight = kwargs.pop("clight", None)
        if self.pn_order > 0 and self.clight is None:
            raise TypeError("'clight' is not defined. Please set the speed of "
                            "light argument 'clight' when using 'pn_order' > 0.")

        self.reporter = kwargs.pop("reporter", None)
        self.dumpper = kwargs.pop("dumpper", None)
        self.dump_freq = kwargs.pop("dump_freq", 1)
        if kwargs:
            msg = "{0}.__init__ received unexpected keyword arguments: {1}."
            raise TypeError(msg.format(type(self).__name__,", ".join(kwargs.keys())))


    def drift_n(self, ip, tau):
        """
        Drift operator for Newtonian quantities.
        """
        if ip.n:
            ip.pos += tau * ip.vel
        return ip


    def drift_pn(self, ip, tau):
        """
        Drift operator for post-Newtonian quantities.
        """
        ip = self.drift_n(ip, tau)
        if ip.blackhole.n:
            for obj in ip.objs():
                if obj.n:
                    if hasattr(obj, "evolve_rcom_pn_shift"):
                        obj.evolve_rcom_pn_shift(tau)
        return ip


    def kick_n(self, ip, jp, tau, update_acc):
        """
        Kick operator for Newtonian quantities.
        """
        if ip.n and jp.n:
            if update_acc: ip.update_acc(jp)
            ip.vel += tau * ip.acc
        return ip


    def kick_pn(self, ip, jp, tau):
        """
        Kick operator for post-Newtonian quantities.
        """
        ip = self.kick_n(ip, jp, tau / 2, True)
        if ip.blackhole.n and jp.blackhole.n:
            for (key, obj) in ip.items():
                if obj.n:
                    if hasattr(obj, "evolve_lmom_pn_shift"):
                        obj.evolve_lmom_pn_shift(tau / 2)
                    if hasattr(obj, "evolve_amom_pn_shift"):
                        obj.evolve_amom_pn_shift(tau / 2)
                    if hasattr(obj, "evolve_ke_pn_shift"):
                        obj.evolve_ke_pn_shift(tau / 2)

                    if hasattr(obj, "get_pnacc"):
                        obj.vel += (tau / 2) * obj.pnacc

                        args = (jp.kind[key], self.pn_order, self.clight)
                        obj.pnacc = 2 * obj.get_pnacc(*args) - obj.pnacc

                        obj.vel += (tau / 2) * obj.pnacc

                    if hasattr(obj, "evolve_ke_pn_shift"):
                        obj.evolve_ke_pn_shift(tau / 2)
                    if hasattr(obj, "evolve_amom_pn_shift"):
                        obj.evolve_amom_pn_shift(tau / 2)
                    if hasattr(obj, "evolve_lmom_pn_shift"):
                        obj.evolve_lmom_pn_shift(tau / 2)
        ip = self.kick_n(ip, jp, tau / 2, False)
        return ip


    def drift(self, ip, tau):
        """
        Drift operator.
        """
        if self.pn_order > 0:
            return self.drift_pn(ip, tau)
        return self.drift_n(ip, tau)


    def kick(self, ip, jp, tau, pn, update_acc=True):
        """
        Kick operator.
        """
        if pn and self.pn_order > 0:
            return self.kick_pn(ip, jp, tau)
        return self.kick_n(ip, jp, tau, update_acc)





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

        if self.pn_order > 0: p.update_pnacc(p, self.pn_order, self.clight)

        if self.dumpper:
            self.dumpper.dump_worldline(p)

        self.is_initialized = True


    def finalize(self, t_end):
        logger.info("Finalizing '%s' integrator.", self.method)

        def final_dump(p, tau, worldline):
            slow, fast = self.split(tau, p)

            if slow.n > 0:
                if worldline:
                    worldline.append(slow)

            if fast.n > 0: final_dump(fast, tau / 2, worldline)

        p = self.particles
        tau = self.get_base_tstep(t_end)

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


    ### split

    def split(self, tau, p):
        slow = p[abs(p.tstep) >= tau]
        fast = p[abs(p.tstep) < tau]

        # prevents the occurrence of a slow level with only one particle.
        if slow.n == 1:
            fast.append(slow)
            slow = type(p)()

        if slow.n + fast.n != p.n:
            logger.error("slow.n + fast.n != p.n: %d, %d, %d.", slow.n, fast.n, p.n)

        return slow, fast


    ### join

    def join(self, slow, fast, inplace=True):
        if slow.n == 0:
            return fast
        if fast.n == 0:
            return slow
        if inplace:
            p = slow
        else:
            p = slow.copy()
        p.append(fast)
        return p


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
        self.wl = type(p)()


        if self.method == "sia.dkd21std":
            p = self.dkd21(p, tau, False, False)
        elif self.method == "sia.dkd21shr":
            p = self.dkd21(p, tau, True, False)
        elif self.method == "sia.dkd21hcc":
            p = self.dkd21(p, tau, True, True)
        elif self.method == "sia.dkd22std":
            p = self.dkd22(p, tau, False, False)
        elif self.method == "sia.dkd22shr":
            p = self.dkd22(p, tau, True, False)
        elif self.method == "sia.dkd22hcc":
            p = self.dkd22(p, tau, True, True)
        elif self.method == "sia.dkd43std":
            p = self.dkd43(p, tau, False, False)
        elif self.method == "sia.dkd43shr":
            p = self.dkd43(p, tau, True, False)
        elif self.method == "sia.dkd43hcc":
            p = self.dkd43(p, tau, True, True)
        elif self.method == "sia.dkd44std":
            p = self.dkd44(p, tau, False, False)
        elif self.method == "sia.dkd44shr":
            p = self.dkd44(p, tau, True, False)
        elif self.method == "sia.dkd44hcc":
            p = self.dkd44(p, tau, True, True)
        elif self.method == "sia.dkd45std":
            p = self.dkd45(p, tau, False, False)
        elif self.method == "sia.dkd45shr":
            p = self.dkd45(p, tau, True, False)
        elif self.method == "sia.dkd45hcc":
            p = self.dkd45(p, tau, True, True)
        elif self.method == "sia.dkd46std":
            p = self.dkd46(p, tau, False, False)
        elif self.method == "sia.dkd46shr":
            p = self.dkd46(p, tau, True, False)
        elif self.method == "sia.dkd46hcc":
            p = self.dkd46(p, tau, True, True)
        elif self.method == "sia.dkd67std":
            p = self.dkd67(p, tau, False, False)
        elif self.method == "sia.dkd67shr":
            p = self.dkd67(p, tau, True, False)
        elif self.method == "sia.dkd67hcc":
            p = self.dkd67(p, tau, True, True)
        elif self.method == "sia.dkd69std":
            p = self.dkd69(p, tau, False, False)
        elif self.method == "sia.dkd69shr":
            p = self.dkd69(p, tau, True, False)
        elif self.method == "sia.dkd69hcc":
            p = self.dkd69(p, tau, True, True)
        elif self.method == 0:
            pass
##            p = self.heapq0(p, tau, True)
        else:
            raise ValueError("Unexpected method: {0}".format(self.method))


        if self.dumpper:
            self.dumpper.dump_worldline(self.wl)

        self.time += self.tstep
        self.particles = p


    def kick_sf(self, slow, fast, tau):
        """
        Slow<->Fast Kick operator.
        """
        fast = self.kick(fast, slow, tau / 2, pn=False)
        slow = self.kick(slow, fast, tau / 2, pn=False)

        slow = self.kick(slow, slow, tau, pn=True)

        slow = self.kick(slow, fast, tau / 2, pn=False)
        fast = self.kick(fast, slow, tau / 2, pn=False)

        return slow, fast


    #
    # dkd21[std,shr,hcc] method -- D.K.D
    #

    def dkd21(self, p, tau, update_tstep, split,
                    k0=1.0,
                    d0=0.5):
        if not p.n: return p

        if update_tstep: p.update_tstep(p, self.eta)

        if split:
            slow, fast = self.split(abs(tau), p)
        else:
            if update_tstep: tau = self.get_min_block_tstep(p, tau)
            slow, fast = p, type(p)()

        #
        fast = self.dkd21(fast, d0 * tau / 2, False, True)
        slow = self.drift(slow, d0 * tau)
        fast = self.dkd21(fast, d0 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k0 * tau)

        fast = self.dkd21(fast, d0 * tau / 2, True, True)
        slow = self.drift(slow, d0 * tau)
        fast = self.dkd21(fast, d0 * tau / 2, True, True)
        #

        if slow.n:
            slow.tstep[:] = tau
            slow.time += tau
            slow.nstep += 1
            wp = slow[slow.nstep % self.dump_freq == 0]
            if wp.n: self.wl.append(wp)

        p = self.join(slow, fast)

        return p


    #
    # dkd22[std,shr,hcc] method -- D.K.D.K.D
    #

    def dkd22(self, p, tau, update_tstep, split,
                    k0=0.5,
                    d0=0.1931833275037836,
                    d1=0.6136333449924328):
        if not p.n: return p

        if update_tstep: p.update_tstep(p, self.eta)

        if split:
            slow, fast = self.split(abs(tau), p)
        else:
            if update_tstep: tau = self.get_min_block_tstep(p, tau)
            slow, fast = p, type(p)()

        #
        fast = self.dkd22(fast, d0 * tau / 2, False, True)
        slow = self.drift(slow, d0 * tau)
        fast = self.dkd22(fast, d0 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k0 * tau)

        fast = self.dkd22(fast, d1 * tau / 2, True, True)
        slow = self.drift(slow, d1 * tau)
        fast = self.dkd22(fast, d1 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k0 * tau)

        fast = self.dkd22(fast, d0 * tau / 2, True, True)
        slow = self.drift(slow, d0 * tau)
        fast = self.dkd22(fast, d0 * tau / 2, True, True)
        #

        if slow.n:
            slow.tstep[:] = tau
            slow.time += tau
            slow.nstep += 1
            wp = slow[slow.nstep % self.dump_freq == 0]
            if wp.n: self.wl.append(wp)

        p = self.join(slow, fast)

        return p


    #
    # dkd43[std,shr,hcc] method -- D.K.D.K.D.K.D
    #

    def dkd43(self, p, tau, update_tstep, split,
                    k0=1.3512071919596575,
                    k1=-1.7024143839193150,
                    d0=0.6756035959798288,
                    d1=-0.17560359597982877):
        if not p.n: return p

        if update_tstep: p.update_tstep(p, self.eta)

        if split:
            slow, fast = self.split(abs(tau), p)
        else:
            if update_tstep: tau = self.get_min_block_tstep(p, tau)
            slow, fast = p, type(p)()

        #
        fast = self.dkd43(fast, d0 * tau / 2, False, True)
        slow = self.drift(slow, d0 * tau)
        fast = self.dkd43(fast, d0 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k0 * tau)

        fast = self.dkd43(fast, d1 * tau / 2, True, True)
        slow = self.drift(slow, d1 * tau)
        fast = self.dkd43(fast, d1 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k1 * tau)

        fast = self.dkd43(fast, d1 * tau / 2, True, True)
        slow = self.drift(slow, d1 * tau)
        fast = self.dkd43(fast, d1 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k0 * tau)

        fast = self.dkd43(fast, d0 * tau / 2, True, True)
        slow = self.drift(slow, d0 * tau)
        fast = self.dkd43(fast, d0 * tau / 2, True, True)
        #

        if slow.n:
            slow.tstep[:] = tau
            slow.time += tau
            slow.nstep += 1
            wp = slow[slow.nstep % self.dump_freq == 0]
            if wp.n: self.wl.append(wp)

        p = self.join(slow, fast)

        return p


    #
    # dkd44[std,shr,hcc] method -- D.K.D.K.D.K.D.K.D
    #

    def dkd44(self, p, tau, update_tstep, split,
                    k0=0.7123418310626056,
                    k1=-0.21234183106260562,
                    d0=0.1786178958448091,
                    d1=-0.06626458266981843,
                    d2=0.7752933736500186):
        if not p.n: return p

        if update_tstep: p.update_tstep(p, self.eta)

        if split:
            slow, fast = self.split(abs(tau), p)
        else:
            if update_tstep: tau = self.get_min_block_tstep(p, tau)
            slow, fast = p, type(p)()

        #
        fast = self.dkd44(fast, d0 * tau / 2, False, True)
        slow = self.drift(slow, d0 * tau)
        fast = self.dkd44(fast, d0 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k0 * tau)

        fast = self.dkd44(fast, d1 * tau / 2, True, True)
        slow = self.drift(slow, d1 * tau)
        fast = self.dkd44(fast, d1 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k1 * tau)

        fast = self.dkd44(fast, d2 * tau / 2, True, True)
        slow = self.drift(slow, d2 * tau)
        fast = self.dkd44(fast, d2 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k1 * tau)

        fast = self.dkd44(fast, d1 * tau / 2, True, True)
        slow = self.drift(slow, d1 * tau)
        fast = self.dkd44(fast, d1 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k0 * tau)

        fast = self.dkd44(fast, d0 * tau / 2, True, True)
        slow = self.drift(slow, d0 * tau)
        fast = self.dkd44(fast, d0 * tau / 2, True, True)
        #

        if slow.n:
            slow.tstep[:] = tau
            slow.time += tau
            slow.nstep += 1
            wp = slow[slow.nstep % self.dump_freq == 0]
            if wp.n: self.wl.append(wp)

        p = self.join(slow, fast)

        return p


    #
    # dkd45[std,shr,hcc] method -- D.K.D.K.D.K.D.K.D.K.D
    #

    def dkd45(self, p, tau, update_tstep, split,
                    k0=-0.0844296195070715,
                    k1=0.354900057157426,
                    k2=0.459059124699291,
                    d0=0.2750081212332419,
                    d1=-0.1347950099106792,
                    d2=0.35978688867743724):
        if not p.n: return p

        if update_tstep: p.update_tstep(p, self.eta)

        if split:
            slow, fast = self.split(abs(tau), p)
        else:
            if update_tstep: tau = self.get_min_block_tstep(p, tau)
            slow, fast = p, type(p)()

        #
        fast = self.dkd45(fast, d0 * tau / 2, False, True)
        slow = self.drift(slow, d0 * tau)
        fast = self.dkd45(fast, d0 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k0 * tau)

        fast = self.dkd45(fast, d1 * tau / 2, True, True)
        slow = self.drift(slow, d1 * tau)
        fast = self.dkd45(fast, d1 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k1 * tau)

        fast = self.dkd45(fast, d2 * tau / 2, True, True)
        slow = self.drift(slow, d2 * tau)
        fast = self.dkd45(fast, d2 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k2 * tau)

        fast = self.dkd45(fast, d2 * tau / 2, True, True)
        slow = self.drift(slow, d2 * tau)
        fast = self.dkd45(fast, d2 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k1 * tau)

        fast = self.dkd45(fast, d1 * tau / 2, True, True)
        slow = self.drift(slow, d1 * tau)
        fast = self.dkd45(fast, d1 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k0 * tau)

        fast = self.dkd45(fast, d0 * tau / 2, True, True)
        slow = self.drift(slow, d0 * tau)
        fast = self.dkd45(fast, d0 * tau / 2, True, True)
        #

        if slow.n:
            slow.tstep[:] = tau
            slow.time += tau
            slow.nstep += 1
            wp = slow[slow.nstep % self.dump_freq == 0]
            if wp.n: self.wl.append(wp)

        p = self.join(slow, fast)

        return p


    #
    # dkd46[std,shr,hcc] method -- D.K.D.K.D.K.D.K.D.K.D.K.D
    #

    def dkd46(self, p, tau, update_tstep, split,
                    k0=0.209515106613362,
                    k1=-0.143851773179818,
                    k2=0.434336666566456,
                    d0=0.0792036964311957,
                    d1=0.353172906049774,
                    d2=-0.0420650803577195,
                    d3=0.21937695575349958):
        if not p.n: return p

        if update_tstep: p.update_tstep(p, self.eta)

        if split:
            slow, fast = self.split(abs(tau), p)
        else:
            if update_tstep: tau = self.get_min_block_tstep(p, tau)
            slow, fast = p, type(p)()

        #
        fast = self.dkd46(fast, d0 * tau / 2, False, True)
        slow = self.drift(slow, d0 * tau)
        fast = self.dkd46(fast, d0 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k0 * tau)

        fast = self.dkd46(fast, d1 * tau / 2, True, True)
        slow = self.drift(slow, d1 * tau)
        fast = self.dkd46(fast, d1 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k1 * tau)

        fast = self.dkd46(fast, d2 * tau / 2, True, True)
        slow = self.drift(slow, d2 * tau)
        fast = self.dkd46(fast, d2 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k2 * tau)

        fast = self.dkd46(fast, d3 * tau / 2, True, True)
        slow = self.drift(slow, d3 * tau)
        fast = self.dkd46(fast, d3 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k2 * tau)

        fast = self.dkd46(fast, d2 * tau / 2, True, True)
        slow = self.drift(slow, d2 * tau)
        fast = self.dkd46(fast, d2 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k1 * tau)

        fast = self.dkd46(fast, d1 * tau / 2, True, True)
        slow = self.drift(slow, d1 * tau)
        fast = self.dkd46(fast, d1 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k0 * tau)

        fast = self.dkd46(fast, d0 * tau / 2, True, True)
        slow = self.drift(slow, d0 * tau)
        fast = self.dkd46(fast, d0 * tau / 2, True, True)
        #

        if slow.n:
            slow.tstep[:] = tau
            slow.time += tau
            slow.nstep += 1
            wp = slow[slow.nstep % self.dump_freq == 0]
            if wp.n: self.wl.append(wp)

        p = self.join(slow, fast)

        return p


    #
    # dkd67[std,shr,hcc] method -- D.K.D.K.D.K.D.K.D.K.D.K.D.K.D
    #

    def dkd67(self, p, tau, update_tstep, split,
                    k0=0.7845136104775573,
                    k1=0.23557321335935813,
                    k2=-1.177679984178871,
                    k3=1.3151863206839112,
                    d0=0.39225680523877865,
                    d1=0.5100434119184577,
                    d2=-0.47105338540975644,
                    d3=0.06875316825252015):
        if not p.n: return p

        if update_tstep: p.update_tstep(p, self.eta)

        if split:
            slow, fast = self.split(abs(tau), p)
        else:
            if update_tstep: tau = self.get_min_block_tstep(p, tau)
            slow, fast = p, type(p)()

        #
        fast = self.dkd67(fast, d0 * tau / 2, False, True)
        slow = self.drift(slow, d0 * tau)
        fast = self.dkd67(fast, d0 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k0 * tau)

        fast = self.dkd67(fast, d1 * tau / 2, True, True)
        slow = self.drift(slow, d1 * tau)
        fast = self.dkd67(fast, d1 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k1 * tau)

        fast = self.dkd67(fast, d2 * tau / 2, True, True)
        slow = self.drift(slow, d2 * tau)
        fast = self.dkd67(fast, d2 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k2 * tau)

        fast = self.dkd67(fast, d3 * tau / 2, True, True)
        slow = self.drift(slow, d3 * tau)
        fast = self.dkd67(fast, d3 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k3 * tau)

        fast = self.dkd67(fast, d3 * tau / 2, True, True)
        slow = self.drift(slow, d3 * tau)
        fast = self.dkd67(fast, d3 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k2 * tau)

        fast = self.dkd67(fast, d2 * tau / 2, True, True)
        slow = self.drift(slow, d2 * tau)
        fast = self.dkd67(fast, d2 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k1 * tau)

        fast = self.dkd67(fast, d1 * tau / 2, True, True)
        slow = self.drift(slow, d1 * tau)
        fast = self.dkd67(fast, d1 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k0 * tau)

        fast = self.dkd67(fast, d0 * tau / 2, True, True)
        slow = self.drift(slow, d0 * tau)
        fast = self.dkd67(fast, d0 * tau / 2, True, True)
        #

        if slow.n:
            slow.tstep[:] = tau
            slow.time += tau
            slow.nstep += 1
            wp = slow[slow.nstep % self.dump_freq == 0]
            if wp.n: self.wl.append(wp)

        p = self.join(slow, fast)

        return p


    #
    # dkd69[std,shr,hcc] method -- D.K.D.K.D.K.D.K.D.K.D.K.D.K.D.K.D.K.D
    #

    def dkd69(self, p, tau, update_tstep, split,
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
        if not p.n: return p

        if update_tstep: p.update_tstep(p, self.eta)

        if split:
            slow, fast = self.split(abs(tau), p)
        else:
            if update_tstep: tau = self.get_min_block_tstep(p, tau)
            slow, fast = p, type(p)()

        #
        fast = self.dkd69(fast, d0 * tau / 2, False, True)
        slow = self.drift(slow, d0 * tau)
        fast = self.dkd69(fast, d0 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k0 * tau)

        fast = self.dkd69(fast, d1 * tau / 2, True, True)
        slow = self.drift(slow, d1 * tau)
        fast = self.dkd69(fast, d1 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k1 * tau)

        fast = self.dkd69(fast, d2 * tau / 2, True, True)
        slow = self.drift(slow, d2 * tau)
        fast = self.dkd69(fast, d2 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k2 * tau)

        fast = self.dkd69(fast, d3 * tau / 2, True, True)
        slow = self.drift(slow, d3 * tau)
        fast = self.dkd69(fast, d3 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k3 * tau)

        fast = self.dkd69(fast, d4 * tau / 2, True, True)
        slow = self.drift(slow, d4 * tau)
        fast = self.dkd69(fast, d4 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k4 * tau)

        fast = self.dkd69(fast, d4 * tau / 2, True, True)
        slow = self.drift(slow, d4 * tau)
        fast = self.dkd69(fast, d4 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k3 * tau)

        fast = self.dkd69(fast, d3 * tau / 2, True, True)
        slow = self.drift(slow, d3 * tau)
        fast = self.dkd69(fast, d3 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k2 * tau)

        fast = self.dkd69(fast, d2 * tau / 2, True, True)
        slow = self.drift(slow, d2 * tau)
        fast = self.dkd69(fast, d2 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k1 * tau)

        fast = self.dkd69(fast, d1 * tau / 2, True, True)
        slow = self.drift(slow, d1 * tau)
        fast = self.dkd69(fast, d1 * tau / 2, True, True)

        slow, fast = self.kick_sf(slow, fast, k0 * tau)

        fast = self.dkd69(fast, d0 * tau / 2, True, True)
        slow = self.drift(slow, d0 * tau)
        fast = self.dkd69(fast, d0 * tau / 2, True, True)
        #

        if slow.n:
            slow.tstep[:] = tau
            slow.time += tau
            slow.nstep += 1
            wp = slow[slow.nstep % self.dump_freq == 0]
            if wp.n: self.wl.append(wp)

        p = self.join(slow, fast)

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
            slow, fast, indexing = self.split(dt, p)
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
            if update_tstep: p.update_tstep(p, self.eta)      # True/False
            if update_tstep: p.update_phi(p)      # True/False
            slow, fast, indexing = self.split(dt/2, p)

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
#            slow, fast, indexing = self.split(tau/2, p)
#
#            if fast.n > 0:
#                tnext += tau/2
#                heapq.heappush(p, (tnext, tau/2, p, slow, fast, indexing))
#            else:
#                tnext += tau
#                heapq.heappush(p, (tnext, tau, p, slow, fast, indexing))



########## end of file ##########
