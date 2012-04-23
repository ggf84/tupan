#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import logging
import numpy as np
from .leapfrog import LeapFrog
from ..lib.utils.timing import timings


__all__ = ["HTS"]

logger = logging.getLogger(__name__)


class HTS(LeapFrog):
    """

    """
    def __init__(self, eta, time, particles, **kwargs):
        self.meth_type = 0
        super(HTS, self).__init__(eta, time, particles, **kwargs)


    def get_max_block_tstep(self, p):
        max_tstep = p.max_dt_next()

        power = int(np.log2(max_tstep) - 1)
        max_block_tstep = 2.0**power

        if (self.time+max_block_tstep)%(max_block_tstep) != 0:
            max_block_tstep /= 2

        return max_block_tstep


    @timings
    def init_for_integration(self, t_end):
        logger.info("Initializing for integration.")

        p = self.particles

        p.update_n()
        p.update_acc(p)
        p.update_pnacc(p)
        p.update_timestep(p, self.eta)

        tau = 1.0/4
#        tau = p.max_dt_next()
#        tau = p.mean_dt_next()
#        tau = p.harmonic_mean_dt_next()
#        self.tstep = tau if self.time+tau < t_end else t_end-self.time

#        tau = self.get_max_block_tstep(p)
        self.tstep = tau

        self.snap_counter = {}
        self.is_initialized = True


    @timings
    def dkd(self, slow, fast, tau):
        """

        """
        self.drift(slow, tau / 2)
        if fast.n > 0: self.kick(fast, slow, tau)
        self.kick(slow, slow, tau)
        if fast.n > 0: self.kick(slow, fast, tau)
        self.drift(slow, tau / 2)


    @timings
    def step(self, t_end):
        """

        """
        if not self.is_initialized:
            self.init_for_integration(t_end)

        tau = self.tstep
        p = self.particles
        self.level = 0

        if self.meth_type == 0:
            self.meth0(p, tau, True)
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

        tau = 1.0/4
#        tau = p.max_dt_next()
#        tau = p.mean_dt_next()
#        tau = p.harmonic_mean_dt_next()
#        self.tstep = tau if self.time+tau < t_end else t_end-self.time

#        tau = self.get_max_block_tstep(p)
        self.tstep = tau


    ### meth0

    def meth0(self, p, tau, update_timestep):
        self.level += 1
#        if update_timestep: p.update_timestep(p, self.eta)      # False/True
        slow, fast, indexing = self.split(tau, p)

        if slow.n > 0: slow.set_dt_next(tau)

        if self.dumpper and slow.n > 0:
            if not self.level in self.snap_counter:
                self.snap_counter[self.level] = self.snap_freq
            self.snap_counter[self.level] += 1
            if (self.snap_counter[self.level] >= self.snap_freq):
                self.snap_counter[self.level] -= self.snap_freq
                self.dumpper.dump(slow)

        if fast.n == 0: self.time += tau / 2
        if fast.n > 0: self.meth0(fast, tau / 2, True)
        if slow.n > 0: self.dkd(slow, fast, tau)
        if fast.n > 0: self.meth0(fast, tau / 2, False)
        if fast.n == 0: self.time += tau / 2

        if slow.n > 0: slow.set_dt_prev(tau)

        self.merge(p, slow, fast, indexing)
        if update_timestep: p.update_timestep(p, self.eta)      # True/False

        self.level -= 1


    ### meth1

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

    def split(self, tau, p):
        slow = p.__class__()
        fast = p.__class__()
        indexing = {}
        for (key, obj) in p.items():
            if obj:
                is_fast = obj.dt_next < tau
                is_slow = ~is_fast
                slow[key] = obj[is_slow]
                fast[key] = obj[is_fast]
                indexing[key] = {'is_slow': is_slow, 'is_fast': is_fast}
            else:
                indexing[key] = {'is_slow': None, 'is_fast': None}

        # prevents the occurrence of a slow level with only one particle.
        slow.update_n()
        if slow.n == 1:
            for (key, obj) in p.items():
                if obj:
                    is_slow = indexing[key]['is_slow']
                    is_fast = indexing[key]['is_fast']
                    is_slow[is_slow] = False
                    is_fast[~is_fast] = True
                    slow[key] = obj[is_slow]
                    fast[key] = obj[is_fast]

        # prevents the occurrence of a fast level with only one particle.
        fast.update_n()
        if fast.n == 1:
            for (key, obj) in p.items():
                if obj:
                    is_slow = indexing[key]['is_slow']
                    is_fast = indexing[key]['is_fast']
                    is_fast[is_fast] = False
                    is_slow[~is_slow] = True
                    slow[key] = obj[is_slow]
                    fast[key] = obj[is_fast]

        fast.update_n()
        slow.update_n()

        if fast.n == 1: logger.error("fast level contains only *one* particle.")
        if slow.n == 1: logger.error("slow level contains only *one* particle.")

        return slow, fast, indexing


    ### merge

    def merge(self, p, slow, fast, indexing):
        # XXX: known bug: numpy fancy indexing in 'split' method returns a copy
        #                 but we want a view. That's why this 'merge' method
        #                 is needed so that after play with slow/fast returned
        #                 from 'split' we should inform p about the changes.
        for (key, obj) in p.items():
            if obj:
                if indexing[key]['is_slow'] is not None:
                    obj.data[indexing[key]['is_slow']] = slow[key].data

                if indexing[key]['is_fast'] is not None:
                    obj.data[indexing[key]['is_fast']] = fast[key].data

        p.update_n()


########## end of file ##########
