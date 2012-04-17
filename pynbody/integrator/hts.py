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
    def __init__(self, eta, time, particles):
        self.meth_type = 0
        super(HTS, self).__init__(eta, time, particles)


    def init_for_integration(self):
        p = self.particles

        p.update_acc(p)
        p.update_pnacc(p)
        p.set_dt_prev()
        p.update_timestep(p, self.eta)
        p.set_dt_next()

        self.tstep = self.eta   # unsused: provisionally sets tstep as eta


    @timings
    def dkd(self, slow, fast, tau):
        """

        """
        self.drift(slow, tau / 2)
        if fast.get_nbody() > 0: self.kick(fast, slow, tau)
        self.kick(slow, slow, tau)
        if fast.get_nbody() > 0: self.kick(slow, fast, tau)
        self.drift(slow, tau / 2)


    @timings
    def step(self, t_end):
        """

        """
        tau = 1.0 / 8
        tau = tau if self.time+tau < t_end else t_end-self.time

        if self.meth_type == 0:
            self.meth0(self.particles, tau, True)
        elif self.meth_type == 1:
            self.meth1(self.particles, tau, True)
        elif self.meth_type == 2:
            self.meth2(self.particles, tau, True)
        elif self.meth_type == 3:
            self.meth3(self.particles, tau, True)
        elif self.meth_type == 4:
            self.meth4(self.particles, tau, True)
        elif self.meth_type == 5:
            self.meth5(self.particles, tau, True)
        elif self.meth_type == 6:
            self.meth6(self.particles, tau, True)
        elif self.meth_type == 7:
            self.meth7(self.particles, tau, True)
        elif self.meth_type == 8:
            self.meth8(self.particles, tau, True)
        elif self.meth_type == 9:
            self.meth9(self.particles, tau, True)
        else:
            raise ValueError("Unexpected HTS method type.")


    ### meth0

    def meth0(self, p, tau, update_timestep):
#        if update_timestep: p.update_timestep(p, self.eta)      # False/True
        slow, fast, indexing = self.split(tau, p, update_timestep)

        if fast.get_nbody() == 0: self.time += tau / 2
        if fast.get_nbody() > 0: self.meth0(fast, tau / 2, True)
        if slow.get_nbody() > 0: self.dkd(slow, fast, tau)
        if fast.get_nbody() > 0: self.meth0(fast, tau / 2, False)
        if fast.get_nbody() == 0: self.time += tau / 2

        self.merge(p, slow, fast, indexing, update_timestep)
        if update_timestep: p.update_timestep(p, self.eta)      # True/False


    ### meth1

    def meth1(self, p, tau, update_timestep):
        slow, fast, indexing = self.split(tau, p, update_timestep)

        if fast.get_nbody() == 0: self.time += tau / 2
        if slow.get_nbody() > 0: self.dkd(slow, fast, tau / 2)
        if fast.get_nbody() > 0: self.meth1(fast, tau / 2, True)
        if fast.get_nbody() > 0: self.meth1(fast, tau / 2, True)
        if slow.get_nbody() > 0: self.dkd(slow, fast, tau / 2)
        if fast.get_nbody() == 0: self.time += tau / 2

        self.merge(p, slow, fast, indexing)


    ### meth2

    def meth2(self, p, tau, update_timestep):
        slow, fast, indexing = self.split(tau, p, update_timestep)

        if fast.get_nbody() == 0: self.time += tau / 2
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau / 2)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau / 2)
        if fast.get_nbody() > 0: self.meth2(fast, tau / 2, True)
        if slow.get_nbody() > 0: self.dkd(slow, fast, tau)
        if fast.get_nbody() > 0: self.meth2(fast, tau / 2, True)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau / 2)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau / 2)
        if fast.get_nbody() == 0: self.time += tau / 2

        self.merge(p, slow, fast, indexing)


    ### meth3

    def meth3(self, p, tau, update_timestep):  # This method does not conserves the center of mass.
        slow, fast, indexing = self.split(tau, p, update_timestep)

        if fast.get_nbody() == 0: self.time += tau / 2
        if fast.get_nbody() > 0: self.meth3(fast, tau / 2, False)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau / 2)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau / 2)
        if slow.get_nbody() > 0: self.dkd(slow, fast, tau)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau / 2)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau / 2)
        if fast.get_nbody() > 0: self.meth3(fast, tau / 2, True)
        if fast.get_nbody() == 0: self.time += tau / 2

        self.merge(p, slow, fast, indexing)


    ### meth4

    def meth4(self, p, tau, update_timestep):
        slow, fast, indexing = self.split(tau, p, update_timestep)

        if fast.get_nbody() == 0: self.time += tau / 2
        if slow.get_nbody() > 0: self.dkd(slow, fast, tau / 2)
        if fast.get_nbody() > 0: self.meth4(fast, tau / 2, False)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau)
        if fast.get_nbody() > 0: self.meth4(fast, tau / 2, True)
        if slow.get_nbody() > 0: self.dkd(slow, fast, tau / 2)
        if fast.get_nbody() == 0: self.time += tau / 2

        self.merge(p, slow, fast, indexing)


    ### meth5

    def meth5(self, p, tau, update_timestep):
        slow, fast, indexing = self.split(tau, p, update_timestep)

        if fast.get_nbody() == 0: self.time += tau / 2
        if fast.get_nbody() > 0: self.meth5(fast, tau / 2, False)
        if slow.get_nbody() > 0: self.dkd(slow, fast, tau / 2)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau)
        if slow.get_nbody() > 0: self.dkd(slow, fast, tau / 2)
        if fast.get_nbody() > 0: self.meth5(fast, tau / 2, True)
        if fast.get_nbody() == 0: self.time += tau / 2

        self.merge(p, slow, fast, indexing)


    ### meth6

    def meth6(self, p, tau, update_timestep):
        slow, fast, indexing = self.split(tau, p, update_timestep)

        if fast.get_nbody() == 0: self.time += tau / 2
        if slow.get_nbody() > 0: self.drift(slow, tau / 2)
        if fast.get_nbody() > 0: self.meth6(fast, tau / 2, False)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau)
        if slow.get_nbody() > 0: self.kick(slow, slow, tau)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau)
        if fast.get_nbody() > 0: self.meth6(fast, tau / 2, True)
        if slow.get_nbody() > 0: self.drift(slow, tau / 2)
        if fast.get_nbody() == 0: self.time += tau / 2

        self.merge(p, slow, fast, indexing)


    ### meth7

    def meth7(self, p, tau, update_timestep):
        slow, fast, indexing = self.split(tau, p, update_timestep)

        if fast.get_nbody() == 0: self.time += tau / 2
        if slow.get_nbody() > 0: self.drift(slow, tau / 2)
        if slow.get_nbody() > 0: self.kick(slow, slow, tau / 2)
        if fast.get_nbody() > 0: self.meth7(fast, tau / 2, False)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau)
        if fast.get_nbody() > 0: self.meth7(fast, tau / 2, True)
        if slow.get_nbody() > 0: self.kick(slow, slow, tau / 2)
        if slow.get_nbody() > 0: self.drift(slow, tau / 2)
        if fast.get_nbody() == 0: self.time += tau / 2

        self.merge(p, slow, fast, indexing)


    ### meth8

    def meth8(self, p, tau, update_timestep):
        slow, fast, indexing = self.split(tau, p, update_timestep)

        if fast.get_nbody() == 0: self.time += tau / 2
        if slow.get_nbody() > 0: self.drift(slow, tau / 2)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau / 2)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau / 2)
        if fast.get_nbody() > 0: self.meth8(fast, tau / 2, True)
        if slow.get_nbody() > 0: self.kick(slow, slow, tau)
        if fast.get_nbody() > 0: self.meth8(fast, tau / 2, True)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau / 2)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau / 2)
        if slow.get_nbody() > 0: self.drift(slow, tau / 2)
        if fast.get_nbody() == 0: self.time += tau / 2

        self.merge(p, slow, fast, indexing)


    ### meth9

    def meth9(self, p, tau, update_timestep):
        slow, fast, indexing = self.split(tau, p, update_timestep)

        if fast.get_nbody() == 0: self.time += tau / 2
        if slow.get_nbody() > 0: self.drift(slow, tau / 2)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau / 2)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau / 2)
        if slow.get_nbody() > 0: self.kick(slow, slow, tau / 2)
        if fast.get_nbody() > 0: self.meth9(fast, tau / 2, True)
        if fast.get_nbody() > 0: self.meth9(fast, tau / 2, True)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau / 2)
        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau / 2)
        if slow.get_nbody() > 0: self.kick(slow, slow, tau / 2)
        if slow.get_nbody() > 0: self.drift(slow, tau / 2)
        if fast.get_nbody() == 0: self.time += tau / 2

        self.merge(p, slow, fast, indexing)


    ### split

    def split(self, tau, p, update_timestep):
#        if update_timestep: p.update_timestep(p, self.eta)

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
        if slow.get_nbody() == 1:
            for (key, obj) in p.items():
                if obj:
                    is_slow = indexing[key]['is_slow']
                    is_fast = indexing[key]['is_fast']
                    is_slow[is_slow] = False
                    is_fast[~is_fast] = True
                    slow[key] = obj[is_slow]
                    fast[key] = obj[is_fast]

        # prevents the occurrence of a fast level with only one particle.
        if fast.get_nbody() == 1:
            for (key, obj) in p.items():
                if obj:
                    is_slow = indexing[key]['is_slow']
                    is_fast = indexing[key]['is_fast']
                    is_fast[is_fast] = False
                    is_slow[~is_slow] = True
                    slow[key] = obj[is_slow]
                    fast[key] = obj[is_fast]

        if fast.get_nbody() == 1: logger.error("fast level contains only *one* particle.")
        if slow.get_nbody() == 1: logger.error("slow level contains only *one* particle.")

        return slow, fast, indexing


    ### merge

    def merge(self, p, slow, fast, indexing, update_timestep):
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

#        if update_timestep: p.update_timestep(p, self.eta)

########## end of file ##########
