#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import logging
import numpy as np
from ..lib.utils.timing import timings


__all__ = ["LeapFrog"]

logger = logging.getLogger(__name__)


class LeapFrog(object):
    """

    """
    def __init__(self, eta, time, particles):
        self.eta = eta
        self.time = time
        particles.update_acctstep(particles, eta)
        self.particles = particles
        self.tstep = self.get_min_block_tstep()
        self.n2_sum = 0


    def get_min_block_tstep(self):
        min_tstep = 1.0
        for (key, obj) in self.particles.items():
            if obj:
                min_tstep = min(min_tstep, obj.tstep.min())

        power = int(np.log2(min_tstep) - 1)
        min_tstep = 2.0**power

        if (self.time+min_tstep)%(min_tstep) != 0:
            min_tstep /= 2

        return min_tstep


    @timings
    def drift(self, ip, tau):
        """

        """
        for (key, obj) in ip.items():
            if obj:
                if hasattr(obj, "evolve_position"):
                    obj.evolve_position(tau)
                if hasattr(obj, "evolve_center_of_mass_position_correction_due_to_pnterms"):
                    obj.evolve_center_of_mass_position_correction_due_to_pnterms(tau)


    @timings
    def forceDKD(self, ip, jp):
        """

        """
        prev_acc = {}
        prev_pnacc = {}
        for (key, obj) in ip.items():
            if obj:
                if hasattr(obj, "acc"):
                    prev_acc[key] = obj.acc.copy()
                if hasattr(obj, "pnacc"):
                    prev_pnacc[key] = obj.pnacc.copy()

        ip.update_acc(jp)

        ni = ip.get_nbody()
        nj = jp.get_nbody()
        self.n2_sum += ni*nj
        ntot = self.particles.get_nbody()
#        if ni == ntot and nj == ntot:
#            print(ni, nj, self.n2_sum)

        for (key, obj) in ip.items():
            if obj:
                if hasattr(obj, "acc"):
                    obj.acc[:] = 2 * obj.acc - prev_acc[key]
                if hasattr(obj, "pnacc"):
                    obj.pnacc[:] = 2 * obj.pnacc - prev_pnacc[key]


    @timings
    def kick(self, ip, jp, tau):
        """

        """
        for (key, obj) in ip.items():
            if obj:
                if hasattr(obj, "evolve_linear_momentum_correction_due_to_pnterms"):
                    obj.evolve_linear_momentum_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_angular_momentum_correction_due_to_pnterms"):
                    obj.evolve_angular_momentum_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_center_of_mass_velocity_correction_due_to_pnterms"):
                    obj.evolve_center_of_mass_velocity_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_energy_correction_due_to_pnterms"):
                    obj.evolve_energy_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_velocity_correction_due_to_pnterms"):
                    obj.evolve_velocity_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_velocity"):
                    obj.evolve_velocity(tau / 2)

        self.forceDKD(ip, jp)

        for (key, obj) in ip.items():
            if obj:
                if hasattr(obj, "evolve_velocity"):
                    obj.evolve_velocity(tau / 2)
                if hasattr(obj, "evolve_velocity_correction_due_to_pnterms"):
                    obj.evolve_velocity_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_energy_correction_due_to_pnterms"):
                    obj.evolve_energy_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_center_of_mass_velocity_correction_due_to_pnterms"):
                    obj.evolve_center_of_mass_velocity_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_angular_momentum_correction_due_to_pnterms"):
                    obj.evolve_angular_momentum_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_linear_momentum_correction_due_to_pnterms"):
                    obj.evolve_linear_momentum_correction_due_to_pnterms(tau / 2)


    @timings
    def stepDKD(self, slow, fast, tau):
        """

        """
        self.drift(slow, tau / 2)
        if fast.get_nbody() > 0: self.kick(fast, slow, tau)
        self.kick(slow, slow, tau)
        if fast.get_nbody() > 0: self.kick(slow, fast, tau)
        self.drift(slow, tau / 2)


    @timings
    def step(self):
        """

        """
#        self.particles.update_tstep(self.particles, self.eta)
#        self.tstep = self.get_min_block_tstep()
#        self.time += self.tstep / 2
#        self.stepDKD(self.particles, self.particles.__class__(), self.tstep)
#        self.time += self.tstep / 2

        tau = 1.0 / 8
        self.rstep(self.particles, tau, True)


    def split(self, tau, p):
        slow = p.__class__()
        fast = p.__class__()
        indexing = {}
        for (key, obj) in p.items():
            if obj:
                is_fast = obj.tstep < tau
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

        return slow, fast, indexing


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


    def rstep(self, p, tau, update_tstep):
        if update_tstep: p.update_tstep(p, self.eta)
        slow, fast, indexing = self.split(tau, p)

        if fast.get_nbody() == 1: logger.error("fast level contains only *one* particle.")
        if slow.get_nbody() == 1: logger.error("slow level contains only *one* particle.")

        # meth 0:
        if fast.get_nbody() == 0: self.time += tau / 2
        if fast.get_nbody() > 0: self.rstep(fast, tau / 2, False)
        if slow.get_nbody() > 0: self.stepDKD(slow, fast, tau)
        if fast.get_nbody() > 0: self.rstep(fast, tau / 2, True)
        if fast.get_nbody() == 0: self.time += tau / 2

#        # meth 1:
#        if fast.get_nbody() == 0: self.time += tau / 2
#        if slow.get_nbody() > 0: self.stepDKD(slow, fast, tau / 2)
#        if fast.get_nbody() > 0: self.rstep(fast, tau / 2, True)
#        if fast.get_nbody() > 0: self.rstep(fast, tau / 2, True)
#        if slow.get_nbody() > 0: self.stepDKD(slow, fast, tau / 2)
#        if fast.get_nbody() == 0: self.time += tau / 2

#        # meth 2:
#        if fast.get_nbody() == 0: self.time += tau / 2
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau / 2)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau / 2)
#        if fast.get_nbody() > 0: self.rstep(fast, tau / 2, True)
#        if slow.get_nbody() > 0: self.stepDKD(slow, fast, tau)
#        if fast.get_nbody() > 0: self.rstep(fast, tau / 2, True)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau / 2)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau / 2)
#        if fast.get_nbody() == 0: self.time += tau / 2

#        # meth 3 (CoM is not conserved):
#        if fast.get_nbody() == 0: self.time += tau / 2
#        if fast.get_nbody() > 0: self.rstep(fast, tau / 2, False)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau / 2)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau / 2)
#        if slow.get_nbody() > 0: self.stepDKD(slow, fast, tau)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau / 2)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau / 2)
#        if fast.get_nbody() > 0: self.rstep(fast, tau / 2, True)
#        if fast.get_nbody() == 0: self.time += tau / 2

#        # meth 4:
#        if fast.get_nbody() == 0: self.time += tau / 2
#        if slow.get_nbody() > 0: self.stepDKD(slow, fast, tau / 2)
#        if fast.get_nbody() > 0: self.rstep(fast, tau / 2, False)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau)
#        if fast.get_nbody() > 0: self.rstep(fast, tau / 2, True)
#        if slow.get_nbody() > 0: self.stepDKD(slow, fast, tau / 2)
#        if fast.get_nbody() == 0: self.time += tau / 2

#        # meth 5:
#        if fast.get_nbody() == 0: self.time += tau / 2
#        if fast.get_nbody() > 0: self.rstep(fast, tau / 2, False)
#        if slow.get_nbody() > 0: self.stepDKD(slow, fast, tau / 2)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau)
#        if slow.get_nbody() > 0: self.stepDKD(slow, fast, tau / 2)
#        if fast.get_nbody() > 0: self.rstep(fast, tau / 2, True)
#        if fast.get_nbody() == 0: self.time += tau / 2

#        # meth 6:
#        if fast.get_nbody() == 0: self.time += tau / 2
#        if slow.get_nbody() > 0: self.drift(slow, tau / 2)
#        if fast.get_nbody() > 0: self.rstep(fast, tau / 2, False)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau)
#        if slow.get_nbody() > 0: self.kick(slow, slow, tau)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau)
#        if fast.get_nbody() > 0: self.rstep(fast, tau / 2, True)
#        if slow.get_nbody() > 0: self.drift(slow, tau / 2)
#        if fast.get_nbody() == 0: self.time += tau / 2

#        # meth 7:
#        if fast.get_nbody() == 0: self.time += tau / 2
#        if slow.get_nbody() > 0: self.drift(slow, tau / 2)
#        if slow.get_nbody() > 0: self.kick(slow, slow, tau / 2)
#        if fast.get_nbody() > 0: self.rstep(fast, tau / 2, False)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau)
#        if fast.get_nbody() > 0: self.rstep(fast, tau / 2, True)
#        if slow.get_nbody() > 0: self.kick(slow, slow, tau / 2)
#        if slow.get_nbody() > 0: self.drift(slow, tau / 2)
#        if fast.get_nbody() == 0: self.time += tau / 2

#        # meth 8:
#        if fast.get_nbody() == 0: self.time += tau / 2
#        if slow.get_nbody() > 0: self.drift(slow, tau / 2)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau / 2)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau / 2)
#        if fast.get_nbody() > 0: self.rstep(fast, tau / 2, True)
#        if slow.get_nbody() > 0: self.kick(slow, slow, tau)
#        if fast.get_nbody() > 0: self.rstep(fast, tau / 2, True)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau / 2)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau / 2)
#        if slow.get_nbody() > 0: self.drift(slow, tau / 2)
#        if fast.get_nbody() == 0: self.time += tau / 2

#        # meth 9:
#        if fast.get_nbody() == 0: self.time += tau / 2
#        if slow.get_nbody() > 0: self.drift(slow, tau / 2)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau / 2)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau / 2)
#        if slow.get_nbody() > 0: self.kick(slow, slow, tau / 2)
#        if fast.get_nbody() > 0: self.rstep(fast, tau / 2, True)
#        if fast.get_nbody() > 0: self.rstep(fast, tau / 2, True)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau / 2)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau / 2)
#        if slow.get_nbody() > 0: self.kick(slow, slow, tau / 2)
#        if slow.get_nbody() > 0: self.drift(slow, tau / 2)
#        if fast.get_nbody() == 0: self.time += tau / 2


        self.merge(p, slow, fast, indexing)


########## end of file ##########
