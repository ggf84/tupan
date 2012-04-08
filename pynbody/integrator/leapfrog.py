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
        particles.set_acctstep(particles, eta)
        self.particles = particles
        self.tstep = self.get_min_block_tstep()
        self.n2_sum = 0


    def get_min_block_tstep(self):
        min_tstep = 1.0
        for (key, obj) in self.particles.items():
            if obj:
                min_tstep = min(min_tstep, obj.tstep.min())

        power = (np.log2(min_tstep) - 1).astype(np.int)
        min_tstep = 2.0**power

        if (self.time+min_tstep)%(min_tstep) != 0:
            min_tstep /= 2

        return min_tstep


    @timings
    def drift(self, ip, tau):
        """

        """
        for (key, obj) in ip.items():
            if hasattr(obj, "evolve_pos"):
                obj.evolve_pos(tau)
            if hasattr(obj, "evolve_com_pos_jump"):
                obj.evolve_com_pos_jump(tau)


    @timings
    def forceDKD(self, ip, jp):
        """

        """
        prev_acc = {}
        prev_pnacc = {}
        for (key, obj) in ip.items():
            if hasattr(obj, "acc"):
                prev_acc[key] = obj.acc.copy()
            if hasattr(obj, "pnacc"):
                prev_pnacc[key] = obj.pnacc.copy()

        ip.set_acc(jp)

        ni = ip.get_nbody()
        nj = jp.get_nbody()
        self.n2_sum += ni*nj
        ntot = self.particles.get_nbody()
        if ni == ntot and nj == ntot:
            print(ni, nj, self.n2_sum)

        for (key, obj) in ip.items():
            if hasattr(obj, "acc"):
                obj.acc[:] = 2 * obj.acc - prev_acc[key]
            if hasattr(obj, "pnacc"):
                obj.pnacc[:] = 2 * obj.pnacc - prev_pnacc[key]


    @timings
    def kick(self, ip, jp, tau):
        """

        """
        for (key, obj) in ip.iteritems():
            if hasattr(obj, "pnacc"):
                external_force = -(obj.mass * obj.pnacc.T).T
                if hasattr(obj, "evolve_com_vel_jump"):
                    obj.evolve_com_vel_jump(0.5 * tau, external_force)
                if hasattr(obj, "evolve_linmom_jump"):
                    obj.evolve_linmom_jump(0.5 * tau, external_force)
                if hasattr(obj, "evolve_angmom_jump"):
                    obj.evolve_angmom_jump(0.5 * tau, external_force)
                if hasattr(obj, "evolve_energy_jump"):
                    obj.evolve_energy_jump(0.5 * tau, external_force)
            if hasattr(obj, "evolve_vel"):
                obj.evolve_vel(0.5 * tau)

        self.forceDKD(ip, jp)

        for (key, obj) in ip.iteritems():
            if hasattr(obj, "evolve_vel"):
                obj.evolve_vel(0.5 * tau)
            if hasattr(obj, "pnacc"):
                external_force = -(obj.mass * obj.pnacc.T).T
                if hasattr(obj, "evolve_com_vel_jump"):
                    obj.evolve_com_vel_jump(0.5 * tau, external_force)
                if hasattr(obj, "evolve_linmom_jump"):
                    obj.evolve_linmom_jump(0.5 * tau, external_force)
                if hasattr(obj, "evolve_angmom_jump"):
                    obj.evolve_angmom_jump(0.5 * tau, external_force)
                if hasattr(obj, "evolve_energy_jump"):
                    obj.evolve_energy_jump(0.5 * tau, external_force)


    @timings
    def stepDKD(self, slow, fast, tau):
        """

        """
        self.drift(slow, tau/2)
        if fast.get_nbody() > 0: self.kick(fast, slow, tau)
        self.kick(slow, slow, tau)
        if fast.get_nbody() > 0: self.kick(slow, fast, tau)
        self.drift(slow, tau/2)


    @timings
    def step(self):
        """

        """
#        self.particles.set_tstep(self.particles, self.eta)
#        self.tstep = self.get_min_block_tstep()
#        self.time += self.tstep/2
#        self.stepDKD(self.particles, self.particles.__class__(), self.tstep)
#        self.time += self.tstep/2

        tau = 1.0/8
        self.rstep(self.particles, tau, True)





    def split(self, tau, p):
        slow = p.__class__()
        fast = p.__class__()
        indexing = {}
        for (key, obj) in p.items():
            if obj:
                is_fast = obj.tstep < tau
                is_slow = ~is_fast

                # prevents the occurrence of a slow level with only one particle.
                obj_slow = obj[np.where(is_slow)]
                if len(obj_slow) == 1:
                    is_slow[np.where(is_slow)] = False
                    is_fast[np.where(~is_fast)] = True
                    obj_slow = obj[np.where(is_slow)]
                    obj_fast = obj[np.where(is_fast)]

                # prevents the occurrence of a fast level with only one particle.
                obj_fast = obj[np.where(is_fast)]
                if len(obj_fast) == 1:
                    is_fast[np.where(is_fast)] = False
                    is_slow[np.where(~is_slow)] = True
                    obj_fast = obj[np.where(is_fast)]
                    obj_slow = obj[np.where(is_slow)]

                slow[key] = obj_slow   # XXX: known bug: numpy fancy indexing returns a copy
                fast[key] = obj_fast   #      but which we want is a view.
                indexing[key] = {'is_slow': is_slow, 'is_fast': is_fast}
            else:
                indexing[key] = {'is_slow': None, 'is_fast': None}

        return slow, fast, indexing


    def merge(self, p, slow, fast, indexing):
        for (key, obj) in p.items():
            if obj:
                if indexing[key]['is_slow'] is not None:
                    obj._data[np.where(indexing[key]['is_slow'])] = slow[key]._data[:]

                if indexing[key]['is_fast'] is not None:
                    obj._data[np.where(indexing[key]['is_fast'])] = fast[key]._data[:]


    def rstep(self, p, tau, update_tstep):
        if update_tstep: p.set_tstep(p, self.eta)
        slow, fast, indexing = self.split(tau, p)

        if fast.get_nbody() == 1: logger.error("fast level contains only *one* particle.")
        if slow.get_nbody() == 1: logger.error("slow level contains only *one* particle.")

        # meth 0:
        if fast.get_nbody() == 0: self.time += tau/2
        if fast.get_nbody() > 0: self.rstep(fast, tau/2, False)
        if slow.get_nbody() > 0: self.stepDKD(slow, fast, tau)
        if fast.get_nbody() > 0: self.rstep(fast, tau/2, True)
        if fast.get_nbody() == 0: self.time += tau/2

#        # meth 1:
#        if fast.get_nbody() == 0: self.time += tau/2
#        if slow.get_nbody() > 0: self.stepDKD(slow, fast, tau/2)
#        if fast.get_nbody() > 0: self.rstep(fast, tau/2, True)
#        if fast.get_nbody() > 0: self.rstep(fast, tau/2, True)
#        if slow.get_nbody() > 0: self.stepDKD(slow, fast, tau/2)
#        if fast.get_nbody() == 0: self.time += tau/2

#        # meth 2:
#        if fast.get_nbody() == 0: self.time += tau/2
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau/2)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau/2)
#        if fast.get_nbody() > 0: self.rstep(fast, tau/2, True)
#        if slow.get_nbody() > 0: self.stepDKD(slow, fast, tau)
#        if fast.get_nbody() > 0: self.rstep(fast, tau/2, True)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau/2)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau/2)
#        if fast.get_nbody() == 0: self.time += tau/2

#        # meth 3 (CoM is not conserved):
#        if fast.get_nbody() == 0: self.time += tau/2
#        if fast.get_nbody() > 0: self.rstep(fast, tau/2, False)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau/2)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau/2)
#        if slow.get_nbody() > 0: self.stepDKD(slow, fast, tau)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau/2)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau/2)
#        if fast.get_nbody() > 0: self.rstep(fast, tau/2, True)
#        if fast.get_nbody() == 0: self.time += tau/2

#        # meth 4:
#        if fast.get_nbody() == 0: self.time += tau/2
#        if slow.get_nbody() > 0: self.stepDKD(slow, fast, tau/2)
#        if fast.get_nbody() > 0: self.rstep(fast, tau/2, False)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau)
#        if fast.get_nbody() > 0: self.rstep(fast, tau/2, True)
#        if slow.get_nbody() > 0: self.stepDKD(slow, fast, tau/2)
#        if fast.get_nbody() == 0: self.time += tau/2

#        # meth 5:
#        if fast.get_nbody() == 0: self.time += tau/2
#        if fast.get_nbody() > 0: self.rstep(fast, tau/2, False)
#        if slow.get_nbody() > 0: self.stepDKD(slow, fast, tau/2)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau)
#        if slow.get_nbody() > 0: self.stepDKD(slow, fast, tau/2)
#        if fast.get_nbody() > 0: self.rstep(fast, tau/2, True)
#        if fast.get_nbody() == 0: self.time += tau/2

#        # meth 6:
#        if fast.get_nbody() == 0: self.time += tau/2
#        if slow.get_nbody() > 0: self.drift(slow, tau/2)
#        if fast.get_nbody() > 0: self.rstep(fast, tau/2, False)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau)
#        if slow.get_nbody() > 0: self.kick(slow, slow, tau)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau)
#        if fast.get_nbody() > 0: self.rstep(fast, tau/2, True)
#        if slow.get_nbody() > 0: self.drift(slow, tau/2)
#        if fast.get_nbody() == 0: self.time += tau/2

#        # meth 7:
#        if fast.get_nbody() == 0: self.time += tau/2
#        if slow.get_nbody() > 0: self.drift(slow, tau/2)
#        if slow.get_nbody() > 0: self.kick(slow, slow, tau/2)
#        if fast.get_nbody() > 0: self.rstep(fast, tau/2, False)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau)
#        if fast.get_nbody() > 0: self.rstep(fast, tau/2, True)
#        if slow.get_nbody() > 0: self.kick(slow, slow, tau/2)
#        if slow.get_nbody() > 0: self.drift(slow, tau/2)
#        if fast.get_nbody() == 0: self.time += tau/2

#        # meth 8:
#        if fast.get_nbody() == 0: self.time += tau/2
#        if slow.get_nbody() > 0: self.drift(slow, tau/2)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau/2)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau/2)
#        if fast.get_nbody() > 0: self.rstep(fast, tau/2, True)
#        if slow.get_nbody() > 0: self.kick(slow, slow, tau)
#        if fast.get_nbody() > 0: self.rstep(fast, tau/2, True)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau/2)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau/2)
#        if slow.get_nbody() > 0: self.drift(slow, tau/2)
#        if fast.get_nbody() == 0: self.time += tau/2

#        # meth 9:
#        if fast.get_nbody() == 0: self.time += tau/2
#        if slow.get_nbody() > 0: self.drift(slow, tau/2)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau/2)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau/2)
#        if slow.get_nbody() > 0: self.kick(slow, slow, tau/2)
#        if fast.get_nbody() > 0: self.rstep(fast, tau/2, True)
#        if fast.get_nbody() > 0: self.rstep(fast, tau/2, True)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(fast, slow, tau/2)
#        if slow.get_nbody() > 0 and fast.get_nbody() > 0: self.kick(slow, fast, tau/2)
#        if slow.get_nbody() > 0: self.kick(slow, slow, tau/2)
#        if slow.get_nbody() > 0: self.drift(slow, tau/2)
#        if fast.get_nbody() == 0: self.time += tau/2


        self.merge(p, slow, fast, indexing)





    # Pickle-related methods

    def __getstate__(self):
        sdict = self.__dict__.copy()
        return sdict

    def __setstate__(self, sdict):
        self.__dict__.update(sdict)
        self.particles = self.particles.copy()


########## end of file ##########
