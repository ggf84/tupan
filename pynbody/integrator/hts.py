#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import logging
import heapq
import numpy as np
from .leapfrog import Base
from ..lib.utils.timing import decallmethods, timings


__all__ = ["HTS"]

logger = logging.getLogger(__name__)


@decallmethods(timings)
class HTS(Base):
    """

    """
    def __init__(self, eta, time, particles, **kwargs):
        super(HTS, self).__init__(eta, time, particles, **kwargs)
        self.meth_type = 0


    def dkd(self, slow, fast, tau):
        """

        """
        self.drift(slow, tau / 2)
        if self.pn_order > 0: self.drift_pn(slow, tau / 2)

        if fast.n > 0: self.kick(fast, slow, tau, True)

        self.kick(slow, slow, tau / 2, True)
        if self.pn_order > 0: self.kick_pn(slow, slow, tau)
        self.kick(slow, slow, tau / 2, False)

        if fast.n > 0: self.kick(slow, fast, tau, True)

        if self.pn_order > 0: self.drift_pn(slow, tau / 2)
        self.drift(slow, tau / 2)

        slow.set_dt_prev(tau)
        slow.update_t_curr(tau)
        slow.update_nstep()
        return slow, fast


    def get_base_tstep(self, t_end):
        tau = self.eta
        self.tstep = tau if self.time + tau <= t_end else t_end - self.time
        return self.tstep


    def initialize(self, t_end):
        logger.info("Initializing '%s' integrator.", self.__class__.__name__.lower())

        p = self.particles

        p.update_acc(p)
        if self.pn_order > 0: p.update_pnacc(p, self.pn_order, self.clight)

        self.is_initialized = True


    def finalize(self, t_end):
        logger.info("Finalizing '%s' integrator.", self.__class__.__name__.lower())

        def final_dump(p, tau, stream):
            slow, fast = self.split(tau, p)

            if slow.n > 0:
                slow.set_dt_next(tau)
                if stream:
                    stream.append(slow)

            if fast.n > 0: final_dump(fast, tau / 2, stream)

        p = self.particles
        tau = self.get_base_tstep(t_end)

        if self.reporter:
            self.reporter.report(self.time, p)
        if self.dumpper:
            stream = p.__class__()
            final_dump(p, tau, stream)
            self.dumpper.dump(stream)


    def evolve_step(self, t_end):
        """

        """
        if not self.is_initialized:
            self.initialize(t_end)

        p = self.particles
        tau = self.get_base_tstep(t_end)

        stream = None
        if self.reporter:
            self.reporter.report(self.time, p)
        if self.dumpper:
            stream = p.__class__()


        if self.meth_type == 0:
            p = self.meth0odr2s1(p, tau, True, stream)
#            p = self.meth0odr4s3(p, tau, True, stream)
#            p = self.meth0odr4s5(p, tau, True, stream)
##            p = self.heapq0(p, tau, True)
        elif self.meth_type == 1:
            p = self.meth1(p, tau, True, stream)
        elif self.meth_type == 2:
            p = self.meth2(p, tau, True, stream)
        elif self.meth_type == 3:
            p = self.meth3(p, tau, True, stream)
        elif self.meth_type == 4:
            p = self.meth4(p, tau, True, stream)
        elif self.meth_type == 5:
            p = self.meth5(p, tau, True, stream)
        elif self.meth_type == 6:
            p = self.meth6(p, tau, True, stream)
        elif self.meth_type == 7:
            p = self.meth7(p, tau, True, stream)
        elif self.meth_type == 8:
            p = self.meth8(p, tau, True, stream)
        elif self.meth_type == 9:
            p = self.meth9(p, tau, True, stream)
        else:
            raise ValueError("Unexpected HTS method type.")


        if stream:
            self.dumpper.dump(stream)

        self.time += tau
        self.particles = p


    ### split

    def split(self, tau, p):
        slow = p.select(lambda x: x >= tau, 'dt_next')
        fast = p.select(lambda x: x < tau, 'dt_next')

        # prevents the occurrence of a slow level with only one particle.
        if slow.n == 1:
            for obj in slow.values():
                if obj.n:
                    fast.append(obj.pop())

        # prevents the occurrence of a fast level with only one particle.
        if fast.n == 1:
            for obj in fast.values():
                if obj.n:
                    slow.append(obj.pop())

        if fast.n == 1: logger.error("fast level contains only *one* particle.")
        if slow.n == 1: logger.error("slow level contains only *one* particle.")
        if slow.n + fast.n != p.n: logger.error("slow.n + fast.n != p.n: %d, %d, %d.", slow.n, fast.n, p.n)

        return slow, fast


    ### join

    def join(self, slow, fast, inplace=True):
        if fast.n == 0:
            return slow
        if slow.n == 0:
            return fast
        if inplace:
            p = slow
        else:
            p = slow.copy()
        p.append(fast)
        return p


    ### meth0

    def meth0odr2s1(self, p, tau, update_tstep, stream):
        if update_tstep: p.update_tstep(p, self.eta)

        slow, fast = self.split(tau, p)

        if slow.n > 0:
            slow.set_dt_next(tau)
            if stream:
                stream.append(slow.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        if fast.n > 0: fast = self.meth0odr2s1(fast, tau / 2, False, stream)
        if slow.n > 0: slow, fast = self.dkd(slow, fast, tau)
        if fast.n > 0: fast = self.meth0odr2s1(fast, tau / 2, True, stream)

        p = self.join(slow, fast)

        return p


    def meth0odr4s3(self, p, tau, update_tstep, stream):
        a = 1.3512071919596575
        b = -1.7024143839193150

        if update_tstep: p.update_tstep(p, self.eta)

        slow, fast = self.split(abs(tau), p)

        if slow.n > 0:
            slow.set_dt_next(tau)
            if stream:
                stream.append(slow.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        if fast.n > 0: fast = self.meth0odr4s3(fast, a*tau / 2, False, stream)
        if slow.n > 0: slow, fast = self.dkd(slow, fast, a*tau)
        if fast.n > 0: fast = self.meth0odr4s3(fast, (a+b)*tau / 2, True, stream)
        if slow.n > 0: slow, fast = self.dkd(slow, fast, b*tau)
        if fast.n > 0: fast = self.meth0odr4s3(fast, (b+a)*tau / 2, True, stream)
        if slow.n > 0: slow, fast = self.dkd(slow, fast, a*tau)
        if fast.n > 0: fast = self.meth0odr4s3(fast, a*tau / 2, True, stream)

        p = self.join(slow, fast)

        return p


    def meth0odr4s5(self, p, tau, update_tstep, stream):
        a = 0.3221375960817983
        b = 0.5413165481700432
        c = -0.7269082885036829

        if update_tstep: p.update_tstep(p, self.eta)

        slow, fast = self.split(abs(tau), p)

        if slow.n > 0:
            slow.set_dt_next(tau)
            if stream:
                stream.append(slow.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        if fast.n > 0: fast = self.meth0odr4s5(fast, a*tau / 2, False, stream)
        if slow.n > 0: slow, fast = self.dkd(slow, fast, a*tau)
        if fast.n > 0: fast = self.meth0odr4s5(fast, (a+b)*tau / 2, True, stream)
        if slow.n > 0: slow, fast = self.dkd(slow, fast, b*tau)
        if fast.n > 0: fast = self.meth0odr4s5(fast, (b+c)*tau / 2, True, stream)
        if slow.n > 0: slow, fast = self.dkd(slow, fast, c*tau)
        if fast.n > 0: fast = self.meth0odr4s5(fast, (c+b)*tau / 2, True, stream)
        if slow.n > 0: slow, fast = self.dkd(slow, fast, b*tau)
        if fast.n > 0: fast = self.meth0odr4s5(fast, (b+a)*tau / 2, True, stream)
        if slow.n > 0: slow, fast = self.dkd(slow, fast, a*tau)
        if fast.n > 0: fast = self.meth0odr4s5(fast, a*tau / 2, True, stream)

        p = self.join(slow, fast)

        return p


    #
    # the following methods are for experimental purposes only.
    #

    ### meth1

    def meth1(self, p, tau, update_tstep, stream):      # This method does not conserves the center of mass.
        if update_tstep: p.update_tstep(p, self.eta)

        slow, fast = self.split(tau, p)

        if slow.n > 0:
            slow.set_dt_next(tau)
            if stream:
                stream.append(slow.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        if slow.n > 0: slow, fast = self.dkd(slow, fast, tau / 2)
        if fast.n > 0: fast = self.meth1(fast, tau / 2, True, stream)
        if fast.n > 0: fast = self.meth1(fast, tau / 2, True, stream)
        if slow.n > 0: slow, fast = self.dkd(slow, fast, tau / 2)

        p = self.join(slow, fast)

        return p


    ### meth2

    def meth2(self, p, tau, update_tstep, stream):
        if update_tstep: p.update_tstep(p, self.eta)

        slow, fast = self.split(tau, p)

        if slow.n > 0:
            slow.set_dt_next(tau)
            if stream:
                stream.append(slow.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau / 2)
        if fast.n > 0: fast = self.meth2(fast, tau / 2, True, stream)
        if slow.n > 0: slow, empty = self.dkd(slow, p.empty, tau)
        if fast.n > 0: fast = self.meth2(fast, tau / 2, True, stream)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau / 2)

        p = self.join(slow, fast)

        return p


    ### meth3

    def meth3(self, p, tau, update_tstep, stream):      # This method does not conserves the center of mass.
        if update_tstep: p.update_tstep(p, self.eta)

        slow, fast = self.split(tau, p)

        if slow.n > 0:
            slow.set_dt_next(tau)
            if stream:
                stream.append(slow.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        if fast.n > 0: fast = self.meth3(fast, tau / 2, False, stream)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau / 2)
        if slow.n > 0: slow, empty = self.dkd(slow, p.empty, tau)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau / 2)
        if fast.n > 0: fast = self.meth3(fast, tau / 2, True, stream)

        p = self.join(slow, fast)

        return p


    ### meth4

    def meth4(self, p, tau, update_tstep, stream):
        if update_tstep: p.update_tstep(p, self.eta)

        slow, fast = self.split(tau, p)

        if slow.n > 0:
            slow.set_dt_next(tau)
            if stream:
                stream.append(slow.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        if slow.n > 0: slow, empty = self.dkd(slow, p.empty, tau / 2)
        if fast.n > 0: fast = self.meth4(fast, tau / 2, False, stream)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau)
        if fast.n > 0: fast = self.meth4(fast, tau / 2, True, stream)
        if slow.n > 0: slow, empty = self.dkd(slow, p.empty, tau / 2)

        p = self.join(slow, fast)

        return p


    ### meth5

    def meth5(self, p, tau, update_tstep, stream):
        if update_tstep: p.update_tstep(p, self.eta)

        slow, fast = self.split(tau, p)

        if slow.n > 0:
            slow.set_dt_next(tau)
            if stream:
                stream.append(slow.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        if fast.n > 0: fast = self.meth5(fast, tau / 2, False, stream)
        if slow.n > 0: slow, empty = self.dkd(slow, p.empty, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau)
        if slow.n > 0: slow, empty = self.dkd(slow, p.empty, tau / 2)
        if fast.n > 0: fast = self.meth5(fast, tau / 2, True, stream)

        p = self.join(slow, fast)

        return p


    ### meth6

    def meth6(self, p, tau, update_tstep, stream):      # results are identical to the meth0
        if update_tstep: p.update_tstep(p, self.eta)

        slow, fast = self.split(tau, p)

        if slow.n > 0:
            slow.set_dt_next(tau)
            if stream:
                stream.append(slow.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        if slow.n > 0: self.drift(slow, tau / 2)
        if fast.n > 0: fast = self.meth6(fast, tau / 2, False, stream)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau)
        if slow.n > 0: self.kick(slow, slow, tau)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau)
        if fast.n > 0: fast = self.meth6(fast, tau / 2, True, stream)
        if slow.n > 0: self.drift(slow, tau / 2)

        p = self.join(slow, fast)

        return p


    ### meth7

    def meth7(self, p, tau, update_tstep, stream):      # results are almost identical to the meth0
        if update_tstep: p.update_tstep(p, self.eta)

        slow, fast = self.split(tau, p)

        if slow.n > 0:
            slow.set_dt_next(tau)
            if stream:
                stream.append(slow.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        if slow.n > 0: self.drift(slow, tau / 2)
        if slow.n > 0: self.kick(slow, slow, tau / 2)
        if fast.n > 0: fast = self.meth7(fast, tau / 2, False, stream)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau)
        if fast.n > 0: fast = self.meth7(fast, tau / 2, True, stream)
        if slow.n > 0: self.kick(slow, slow, tau / 2)
        if slow.n > 0: self.drift(slow, tau / 2)

        p = self.join(slow, fast)

        return p


    ### meth8

    def meth8(self, p, tau, update_tstep, stream):      # This method does not conserves the center of mass.
        if update_tstep: p.update_tstep(p, self.eta)

        slow, fast = self.split(tau, p)

        if slow.n > 0:
            slow.set_dt_next(tau)
            if stream:
                stream.append(slow.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        if slow.n > 0: self.drift(slow, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau / 2)
        if fast.n > 0: fast = self.meth8(fast, tau / 2, True, stream)
        if slow.n > 0: self.kick(slow, slow, tau)
        if fast.n > 0: fast = self.meth8(fast, tau / 2, True, stream)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau / 2)
        if slow.n > 0: self.drift(slow, tau / 2)

        p = self.join(slow, fast)

        return p


    ### meth9

    def meth9(self, p, tau, update_tstep, stream):      # This method does not conserves the center of mass.
        if update_tstep: p.update_tstep(p, self.eta)

        slow, fast = self.split(tau, p)

        if slow.n > 0:
            slow.set_dt_next(tau)
            if stream:
                stream.append(slow.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        if slow.n > 0: self.drift(slow, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau / 2)
        if slow.n > 0: self.kick(slow, slow, tau / 2)
        if fast.n > 0: fast = self.meth9(fast, tau / 2, True, stream)
        if fast.n > 0: fast = self.meth9(fast, tau / 2, True, stream)
        if slow.n > 0: self.kick(slow, slow, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau / 2)
        if slow.n > 0: self.drift(slow, tau / 2)

        p = self.join(slow, fast)

        return p



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
                self.dkd(slow, fast, dt)
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
#            if slow.n > 0: self.dkd(slow, fast, tau)
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
