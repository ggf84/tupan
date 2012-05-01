#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import logging
import heapq
import numpy as np
from .leapfrog import LeapFrog
from ..lib.utils.timing import timings


__all__ = ["HTS"]

logger = logging.getLogger(__name__)


class HTS(LeapFrog):
    """

    """
    @timings
    def __init__(self, eta, time, particles, **kwargs):
        self.meth_type = 0
        super(HTS, self).__init__(eta, time, particles, **kwargs)


    @timings
    def get_max_block_tstep(self, p):
        max_tstep = p.max_dt_next()

        power = int(np.log2(max_tstep) - 1)
        max_block_tstep = 2.0**power

        if (self.time+max_block_tstep)%(max_block_tstep) != 0:
            max_block_tstep /= 2

        return max_block_tstep


    @timings
    def initialize(self, t_end):
        logger.info("Initializing integrator.")

        p = self.particles

        p.update_n()
        p.update_phi(p)
        p.update_acc(p)
        p.update_pnacc(p)
        p.update_timestep(p, self.eta)

#        tau = 1.0/4
#        tau = p.max_dt_next()
#        tau = p.mean_dt_next()
#        tau = p.harmonic_mean_dt_next()
#        self.tstep = tau if self.time+tau < t_end else t_end-self.time

        tau = self.get_max_block_tstep(p)
        self.tstep = tau

        self.snap_counter = {}
        self.is_initialized = True


    @timings
    def finalize(self, t_end):
        logger.info("Finalizing integrator.")

        tau = self.tstep
        p = self.particles

        def final_dump(p, tau):
            slow, fast = self.split(tau, p)

            if slow.n > 0:
                slow.set_dt_next(tau)
                if self.dumpper:
                    self.dumpper.dump(slow)

            if fast.n > 0: final_dump(fast, tau / 2)

        final_dump(p, tau)


    @timings
    def dkd(self, slow, fast, tau):
        """

        """
        self.drift(slow, tau / 2)
        if fast.n > 0: self.kick(fast, slow, tau)
        self.kick(slow, slow, tau)
        if fast.n > 0: self.kick(slow, fast, tau)
        self.drift(slow, tau / 2)

        return slow, fast


    @timings
    def step(self, t_end):
        """

        """
        if not self.is_initialized:
            self.initialize(t_end)

        tau = self.tstep
        p = self.particles
        self.level = -1

        if self.meth_type == 0:
            p = self.meth0(p, tau, True)
#            p = self.heapq0(p, tau, True)
        elif self.meth_type == 1:
            self.meth1(p, tau, True)
        elif self.meth_type == 2:
            self.meth2(p, tau, True)
        elif self.meth_type == 3:
            self.meth3(p, tau, True)
        elif self.meth_type == 4:
            self.meth4(p, tau, True)
        elif self.meth_type == 5:
            self.meth5(p, tau, True)
        elif self.meth_type == 6:
            self.meth6(p, tau, True)
        elif self.meth_type == 7:
            self.meth7(p, tau, True)
        elif self.meth_type == 8:
            self.meth8(p, tau, True)
        elif self.meth_type == 9:
            self.meth9(p, tau, True)
        else:
            raise ValueError("Unexpected HTS method type.")

#        tau = 1.0/4
#        tau = p.max_dt_next()
#        tau = p.mean_dt_next()
#        tau = p.harmonic_mean_dt_next()
#        self.tstep = tau if self.time+tau < t_end else t_end-self.time

        tau = self.get_max_block_tstep(p)
        self.tstep = tau

        self.particles = p


    ### heapq0

    def heapq0(self, p, tau, update_timestep):

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
            if update_timestep: p.update_timestep(p, self.eta)      # True/False
            if update_timestep: p.update_phi(p)      # True/False
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
#            if update_timestep: p.update_timestep(p, self.eta)      # True/False
#
#            slow, fast, indexing = self.split(tau/2, p)
#
#            if fast.n > 0:
#                tnext += tau/2
#                heapq.heappush(p, (tnext, tau/2, p, slow, fast, indexing))
#            else:
#                tnext += tau
#                heapq.heappush(p, (tnext, tau, p, slow, fast, indexing))





    ### meth0

    @timings
    def meth0(self, p, tau, update_timestep):
        self.level += 1
#        if update_timestep: p.update_timestep(p, self.eta)      # False/True
#        if update_timestep: p.update_phi(p)                     # False/True

        slow, fast = self.split(tau, p)

        if slow.n > 0:
            slow.set_dt_next(tau)
            if self.dumpper:
                if not self.level in self.snap_counter:
                    self.snap_counter[self.level] = self.snap_freq
                self.snap_counter[self.level] += 1
                if (self.snap_counter[self.level] >= self.snap_freq):
                    self.snap_counter[self.level] -= self.snap_freq
                    self.dumpper.dump(slow)

        if fast.n == 0: self.time += tau / 2
        if fast.n > 0: fast = self.meth0(fast, tau / 2, True)
        if slow.n > 0: slow, fast = self.dkd(slow, fast, tau)
        if fast.n > 0: fast = self.meth0(fast, tau / 2, False)
        if fast.n == 0: self.time += tau / 2

        if slow.n > 0: slow.set_dt_prev(tau)

        p = self.join(slow, fast)
        if update_timestep: p.update_timestep(p, self.eta)      # True/False
        if update_timestep: p.update_phi(p)                     # True/False

        self.level -= 1

        return p


    ### meth1

    @timings
    def meth1(self, p, tau, update_timestep):
        if update_timestep: p.update_timestep(p, self.eta)
        slow, fast, indexing = self.split(tau, p)

        if fast.n == 0: self.time += tau / 2
        if slow.n > 0: self.dkd(slow, fast, tau / 2)
        if fast.n > 0: self.meth1(fast, tau / 2, True)
        if fast.n > 0: self.meth1(fast, tau / 2, True)
        if slow.n > 0: self.dkd(slow, fast, tau / 2)
        if fast.n == 0: self.time += tau / 2

        self.merge(p, slow, fast, indexing)
#        if update_timestep: p.update_timestep(p, self.eta)


    ### meth2

    @timings
    def meth2(self, p, tau, update_timestep):
        if update_timestep: p.update_timestep(p, self.eta)
        slow, fast, indexing = self.split(tau, p)

        if fast.n == 0: self.time += tau / 2
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau / 2)
        if fast.n > 0: self.meth2(fast, tau / 2, True)
        if slow.n > 0: self.dkd(slow, fast, tau)
        if fast.n > 0: self.meth2(fast, tau / 2, True)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau / 2)
        if fast.n == 0: self.time += tau / 2

        self.merge(p, slow, fast, indexing)
#        if update_timestep: p.update_timestep(p, self.eta)


    ### meth3

    @timings
    def meth3(self, p, tau, update_timestep):  # This method does not conserves the center of mass.
        if update_timestep: p.update_timestep(p, self.eta)
        slow, fast, indexing = self.split(tau, p)

        if fast.n == 0: self.time += tau / 2
        if fast.n > 0: self.meth3(fast, tau / 2, False)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau / 2)
        if slow.n > 0: self.dkd(slow, fast, tau)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau / 2)
        if fast.n > 0: self.meth3(fast, tau / 2, True)
        if fast.n == 0: self.time += tau / 2

        self.merge(p, slow, fast, indexing)
#        if update_timestep: p.update_timestep(p, self.eta)


    ### meth4

    @timings
    def meth4(self, p, tau, update_timestep):
        if update_timestep: p.update_timestep(p, self.eta)
        slow, fast, indexing = self.split(tau, p)

        if fast.n == 0: self.time += tau / 2
        if slow.n > 0: self.dkd(slow, fast, tau / 2)
        if fast.n > 0: self.meth4(fast, tau / 2, False)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau)
        if fast.n > 0: self.meth4(fast, tau / 2, True)
        if slow.n > 0: self.dkd(slow, fast, tau / 2)
        if fast.n == 0: self.time += tau / 2

        self.merge(p, slow, fast, indexing)
#        if update_timestep: p.update_timestep(p, self.eta)


    ### meth5

    @timings
    def meth5(self, p, tau, update_timestep):
        if update_timestep: p.update_timestep(p, self.eta)
        slow, fast, indexing = self.split(tau, p)

        if fast.n == 0: self.time += tau / 2
        if fast.n > 0: self.meth5(fast, tau / 2, False)
        if slow.n > 0: self.dkd(slow, fast, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau)
        if slow.n > 0: self.dkd(slow, fast, tau / 2)
        if fast.n > 0: self.meth5(fast, tau / 2, True)
        if fast.n == 0: self.time += tau / 2

        self.merge(p, slow, fast, indexing)
#        if update_timestep: p.update_timestep(p, self.eta)


    ### meth6

    @timings
    def meth6(self, p, tau, update_timestep):
        if update_timestep: p.update_timestep(p, self.eta)
        slow, fast, indexing = self.split(tau, p)

        if fast.n == 0: self.time += tau / 2
        if slow.n > 0: self.drift(slow, tau / 2)
        if fast.n > 0: self.meth6(fast, tau / 2, False)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau)
        if slow.n > 0: self.kick(slow, slow, tau)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau)
        if fast.n > 0: self.meth6(fast, tau / 2, True)
        if slow.n > 0: self.drift(slow, tau / 2)
        if fast.n == 0: self.time += tau / 2

        self.merge(p, slow, fast, indexing)
#        if update_timestep: p.update_timestep(p, self.eta)


    ### meth7

    @timings
    def meth7(self, p, tau, update_timestep):
        if update_timestep: p.update_timestep(p, self.eta)
        slow, fast, indexing = self.split(tau, p)

        if fast.n == 0: self.time += tau / 2
        if slow.n > 0: self.drift(slow, tau / 2)
        if slow.n > 0: self.kick(slow, slow, tau / 2)
        if fast.n > 0: self.meth7(fast, tau / 2, False)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau)
        if fast.n > 0: self.meth7(fast, tau / 2, True)
        if slow.n > 0: self.kick(slow, slow, tau / 2)
        if slow.n > 0: self.drift(slow, tau / 2)
        if fast.n == 0: self.time += tau / 2

        self.merge(p, slow, fast, indexing)
#        if update_timestep: p.update_timestep(p, self.eta)


    ### meth8

    @timings
    def meth8(self, p, tau, update_timestep):
        if update_timestep: p.update_timestep(p, self.eta)
        slow, fast, indexing = self.split(tau, p)

        if fast.n == 0: self.time += tau / 2
        if slow.n > 0: self.drift(slow, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau / 2)
        if fast.n > 0: self.meth8(fast, tau / 2, True)
        if slow.n > 0: self.kick(slow, slow, tau)
        if fast.n > 0: self.meth8(fast, tau / 2, True)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau / 2)
        if slow.n > 0: self.drift(slow, tau / 2)
        if fast.n == 0: self.time += tau / 2

        self.merge(p, slow, fast, indexing)
#        if update_timestep: p.update_timestep(p, self.eta)


    ### meth9

    @timings
    def meth9(self, p, tau, update_timestep):
        if update_timestep: p.update_timestep(p, self.eta)
        slow, fast, indexing = self.split(tau, p)

        if fast.n == 0: self.time += tau / 2
        if slow.n > 0: self.drift(slow, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau / 2)
        if slow.n > 0: self.kick(slow, slow, tau / 2)
        if fast.n > 0: self.meth9(fast, tau / 2, True)
        if fast.n > 0: self.meth9(fast, tau / 2, True)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau / 2)
        if slow.n > 0: self.kick(slow, slow, tau / 2)
        if slow.n > 0: self.drift(slow, tau / 2)
        if fast.n == 0: self.time += tau / 2

        self.merge(p, slow, fast, indexing)
#        if update_timestep: p.update_timestep(p, self.eta)


    ### split

    @timings
    def split(self, tau, p):
        slow = p.__class__()
        fast = p.__class__()
        for obj in p.values():
            if obj:
                is_fast = obj.dt_next < tau
                is_slow = ~is_fast
                slow.append(obj[is_slow])
                fast.append(obj[is_fast])

        # prevents the occurrence of a slow level with only one particle.
        slow.update_n()
        if slow.n == 1:
            for obj in slow.values():
                if obj:
                    fast.append(obj.pop())

        # prevents the occurrence of a fast level with only one particle.
        fast.update_n()
        if fast.n == 1:
            for obj in fast.values():
                if obj:
                    slow.append(obj.pop())

        fast.update_n()
        slow.update_n()

        if fast.n == 1: logger.error("fast level contains only *one* particle.")
        if slow.n == 1: logger.error("slow level contains only *one* particle.")
        if slow.n + fast.n != p.n: logger.error("slow.n + fast.n != p.n: %d, %d, %d.", slow.n, fast.n, p.n)

        return slow, fast


    ### join

    @timings
    def join(self, slow, fast):
        if fast.n == 0:
            return slow
        if slow.n == 0:
            return fast
        p = slow
        p.append(fast)
        p.update_n()
        return p


########## end of file ##########
