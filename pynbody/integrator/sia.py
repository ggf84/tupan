#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import logging
import heapq
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
            raise TypeError(msg.format(self.__class__.__name__,", ".join(kwargs.keys())))


    def drift(self, ip, tau):
        """
        Drift operator.
        """
        for obj in ip.values():
            if obj.n:
                obj.pos += tau * obj.vel


    def kick(self, ip, jp, tau, update_acc):
        """
        Kick operator.
        """
        if update_acc: ip.update_acc(jp)
        for obj in ip.values():
            if obj.n:
                obj.vel += tau * obj.acc


    def drift_pn(self, ip, tau):
        """
        Drift operator for PN quantities.
        """
        for obj in ip.values():
            if obj.n:
                if hasattr(obj, "evolve_center_of_mass_position_correction_due_to_pnterms"):
                    obj.evolve_center_of_mass_position_correction_due_to_pnterms(tau)


    def kick_pn(self, ip, jp, tau):
        """
        Kick operator for PN quantities.
        """
        if not (ip.blackhole.n and jp.blackhole.n): return
        prev_pnacc = {}
        for (key, obj) in ip.items():
            if obj.n:
                if hasattr(obj, "evolve_linear_momentum_correction_due_to_pnterms"):
                    obj.evolve_linear_momentum_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_angular_momentum_correction_due_to_pnterms"):
                    obj.evolve_angular_momentum_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_energy_correction_due_to_pnterms"):
                    obj.evolve_energy_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_velocity_correction_due_to_pnterms"):
                    obj.evolve_velocity_correction_due_to_pnterms(tau / 2)

                if hasattr(obj, "pnacc"):
                    prev_pnacc[key] = obj.pnacc.copy()

        ip.update_pnacc(jp, self.pn_order, self.clight)

        for (key, obj) in ip.items():
            if obj.n:
                if hasattr(obj, "pnacc"):
                    obj.pnacc = 2 * obj.pnacc - prev_pnacc[key]

                if hasattr(obj, "evolve_velocity_correction_due_to_pnterms"):
                    obj.evolve_velocity_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_energy_correction_due_to_pnterms"):
                    obj.evolve_energy_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_angular_momentum_correction_due_to_pnterms"):
                    obj.evolve_angular_momentum_correction_due_to_pnterms(tau / 2)
                if hasattr(obj, "evolve_linear_momentum_correction_due_to_pnterms"):
                    obj.evolve_linear_momentum_correction_due_to_pnterms(tau / 2)


    def dkd(self, p, tau):
        """
        The Drift-Kick-Drift operator.
        """
        self.drift(p, tau / 2)
        if self.pn_order > 0: self.drift_pn(p, tau / 2)

        self.kick(p, p, tau / 2, True)
        if self.pn_order > 0: self.kick_pn(p, p, tau)
        self.kick(p, p, tau / 2, False)

        if self.pn_order > 0: self.drift_pn(p, tau / 2)
        self.drift(p, tau / 2)
        return p


    def sf_dkd(self, slow, fast, tau):
        """
        The Drift-Kick-Drift operator for hierarchical methods.
        """
        self.drift(slow, tau / 2)
        if self.pn_order > 0: self.drift_pn(slow, tau / 2)

        if fast.n: self.kick(fast, slow, tau, True)

        self.kick(slow, slow, tau / 2, True)
        if self.pn_order > 0: self.kick_pn(slow, slow, tau)
        self.kick(slow, slow, tau / 2, False)

        if fast.n: self.kick(slow, fast, tau, True)

        if self.pn_order > 0: self.drift_pn(slow, tau / 2)
        self.drift(slow, tau / 2)

        return slow, fast




@decallmethods(timings)
class SIA(Base):
    """

    """
    PROVIDED_METHODS = ['sia.dkd21std', 'sia.dkd21shr', 'sia.dkd21hcc',
                        'sia.dkd43std', 'sia.dkd43shr', 'sia.dkd43hcc',
                        'sia.dkd45std', 'sia.dkd45shr', 'sia.dkd45hcc',
                        'sia.dkd67std', 'sia.dkd67shr', 'sia.dkd67hcc',
                        'sia.dkd69std', 'sia.dkd69shr', 'sia.dkd69hcc',
                       ]

    def __init__(self, eta, time, particles, method, **kwargs):
        super(SIA, self).__init__(eta, time, particles, **kwargs)
        self.method = method


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


    def get_min_block_tstep(self, p, tau):
        min_tstep = p.min_dt_next()

        power = int(np.log2(min_tstep) - 1)
        min_block_tstep = 2.0**power

        next_time = self.time + min_block_tstep
        if next_time % min_block_tstep != 0:
            min_block_tstep /= 2

        self.tstep = min_block_tstep if min_block_tstep < tau else tau
        return self.tstep


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


    ### evolve_step

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


        if self.method == "sia.dkd21std":
            p = self.dkd21(p, tau, False, stream)
        elif self.method == "sia.dkd21shr":
            p = self.dkd21(p, tau, True, stream)
        elif self.method == "sia.dkd21hcc":
            p = self.dkd21hcc(p, tau, True, stream)
        elif self.method == "sia.dkd43std":
            p = self.dkd43(p, tau, False, stream)
        elif self.method == "sia.dkd43shr":
            p = self.dkd43(p, tau, True, stream)
        elif self.method == "sia.dkd43hcc":
            p = self.dkd43hcc(p, tau, True, stream)
        elif self.method == "sia.dkd45std":
            p = self.dkd45(p, tau, False, stream)
        elif self.method == "sia.dkd45shr":
            p = self.dkd45(p, tau, True, stream)
        elif self.method == "sia.dkd45hcc":
            p = self.dkd45hcc(p, tau, True, stream)
        elif self.method == "sia.dkd67std":
            p = self.dkd67(p, tau, False, stream)
        elif self.method == "sia.dkd67shr":
            p = self.dkd67(p, tau, True, stream)
        elif self.method == "sia.dkd67hcc":
            p = self.dkd67hcc(p, tau, True, stream)
        elif self.method == "sia.dkd69std":
            p = self.dkd69(p, tau, False, stream)
        elif self.method == "sia.dkd69shr":
            p = self.dkd69(p, tau, True, stream)
        elif self.method == "sia.dkd69hcc":
            p = self.dkd69hcc(p, tau, True, stream)
        elif self.method == 1:
            p = self.meth1(p, tau, True, stream)
        elif self.method == 2:
            p = self.meth2(p, tau, True, stream)
        elif self.method == 3:
            p = self.meth3(p, tau, True, stream)
        elif self.method == 4:
            p = self.meth4(p, tau, True, stream)
        elif self.method == 5:
            p = self.meth5(p, tau, True, stream)
        elif self.method == 6:
            p = self.meth6(p, tau, True, stream)
        elif self.method == 7:
            p = self.meth7(p, tau, True, stream)
        elif self.method == 8:
            p = self.meth8(p, tau, True, stream)
        elif self.method == 9:
            p = self.meth9(p, tau, True, stream)
        elif self.method == 0:
            pass
##            p = self.heapq0(p, tau, True)
        else:
            raise ValueError("Unexpected HTS method type.")


        if stream:
            self.dumpper.dump(stream)

        self.time += self.tstep
        self.particles = p


    #
    # dkd21[std,shr,hcc] methods
    #

    def dkd21(self, p, tau, update_tstep, stream):
        if update_tstep:
            p.update_tstep(p, self.eta)
            tau = self.get_min_block_tstep(p, tau)

        p.set_dt_next(tau)

        if stream:
            stream.append(p.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        p = self.dkd(p, tau)

        p.set_dt_prev(tau)
        p.update_t_curr(tau)
        p.update_nstep()

        return p


    def dkd21hcc(self, p, tau, update_tstep, stream):
        if update_tstep: p.update_tstep(p, self.eta)

        slow, fast = self.split(tau, p)

        if slow.n:
            slow.set_dt_next(tau)
            if stream:
                stream.append(slow.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        if fast.n: fast = self.dkd21hcc(fast, tau / 2, False, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, tau)
        if fast.n: fast = self.dkd21hcc(fast, tau / 2, True, stream)

        if slow.n:
            slow.set_dt_prev(tau)
            slow.update_t_curr(tau)
            slow.update_nstep()

        p = self.join(slow, fast)

        return p


    #
    # dkd43[std,shr,hcc] methods
    #

    def dkd43(self, p, tau, update_tstep, stream,
                    c0=1.3512071919596575,
                    c1=-0.35120719195965755,    # c0+c2
                    c2=-1.7024143839193150):
        if update_tstep:
            p.update_tstep(p, self.eta)
            tau = self.get_min_block_tstep(p, tau)

        p.set_dt_next(tau)

        if stream:
            stream.append(p.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        p = self.dkd(p, c0 * tau)
        p = self.dkd(p, c2 * tau)
        p = self.dkd(p, c0 * tau)

        p.set_dt_prev(tau)
        p.update_t_curr(tau)
        p.update_nstep()

        return p


    def dkd43hcc(self, p, tau, update_tstep, stream,
                       c0=1.3512071919596575,
                       c1=-0.35120719195965755,    # c0+c2
                       c2=-1.7024143839193150):
        if update_tstep: p.update_tstep(p, self.eta)

        slow, fast = self.split(abs(tau), p)

        if slow.n:
            slow.set_dt_next(tau)
            if stream:
                stream.append(slow.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        if fast.n: fast = self.dkd43hcc(fast, c0 * tau / 2, False, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c0 * tau)
        if fast.n: fast = self.dkd43hcc(fast, c1 * tau / 2, True, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c2 * tau)
        if fast.n: fast = self.dkd43hcc(fast, c1 * tau / 2, True, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c0 * tau)
        if fast.n: fast = self.dkd43hcc(fast, c0 * tau / 2, True, stream)

        if slow.n:
            slow.set_dt_prev(tau)
            slow.update_t_curr(tau)
            slow.update_nstep()

        p = self.join(slow, fast)

        return p


    #
    # dkd45[std,shr,hcc] methods
    #

    def dkd45(self, p, tau, update_tstep, stream,
                    c0=0.3221375960817983,
                    c1=0.8634541442518415,      # c0+c2
                    c2=0.5413165481700432,
                    c3=-0.1855917403336398,     # c2+c4
                    c4=-0.7269082885036829):
        if update_tstep:
            p.update_tstep(p, self.eta)
            tau = self.get_min_block_tstep(p, tau)

        p.set_dt_next(tau)

        if stream:
            stream.append(p.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        p = self.dkd(p, c0 * tau)
        p = self.dkd(p, c2 * tau)
        p = self.dkd(p, c4 * tau)
        p = self.dkd(p, c2 * tau)
        p = self.dkd(p, c0 * tau)

        p.set_dt_prev(tau)
        p.update_t_curr(tau)
        p.update_nstep()

        return p


    def dkd45hcc(self, p, tau, update_tstep, stream,
                       c0=0.3221375960817983,
                       c1=0.8634541442518415,      # c0+c2
                       c2=0.5413165481700432,
                       c3=-0.1855917403336398,     # c2+c4
                       c4=-0.7269082885036829):
        if update_tstep: p.update_tstep(p, self.eta)

        slow, fast = self.split(abs(tau), p)

        if slow.n:
            slow.set_dt_next(tau)
            if stream:
                stream.append(slow.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        if fast.n: fast = self.dkd45hcc(fast, c0 * tau / 2, False, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c0 * tau)
        if fast.n: fast = self.dkd45hcc(fast, c1 * tau / 2, True, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c2 * tau)
        if fast.n: fast = self.dkd45hcc(fast, c3 * tau / 2, True, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c4 * tau)
        if fast.n: fast = self.dkd45hcc(fast, c3 * tau / 2, True, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c2 * tau)
        if fast.n: fast = self.dkd45hcc(fast, c1 * tau / 2, True, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c0 * tau)
        if fast.n: fast = self.dkd45hcc(fast, c0 * tau / 2, True, stream)

        if slow.n:
            slow.set_dt_prev(tau)
            slow.update_t_curr(tau)
            slow.update_nstep()

        p = self.join(slow, fast)

        return p


    #
    # dkd67[std,shr,hcc] methods
    #

    def dkd67(self, p, tau, update_tstep, stream,
                    c0=0.7845136104775573,
                    c1=1.0200868238369154,     # c0+c2
                    c2=0.23557321335935813,
                    c3=-0.9421067708195129,    # c2+c4
                    c4=-1.177679984178871,
                    c5=0.13750633650504018,    # c4+c6
                    c6=1.3151863206839112):
        if update_tstep:
            p.update_tstep(p, self.eta)
            tau = self.get_min_block_tstep(p, tau)

        p.set_dt_next(tau)

        if stream:
            stream.append(p.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        p = self.dkd(p, c0 * tau)
        p = self.dkd(p, c2 * tau)
        p = self.dkd(p, c4 * tau)
        p = self.dkd(p, c6 * tau)
        p = self.dkd(p, c4 * tau)
        p = self.dkd(p, c2 * tau)
        p = self.dkd(p, c0 * tau)

        p.set_dt_prev(tau)
        p.update_t_curr(tau)
        p.update_nstep()

        return p


    def dkd67hcc(self, p, tau, update_tstep, stream,
                       c0=0.7845136104775573,
                       c1=1.0200868238369154,     # c0+c2
                       c2=0.23557321335935813,
                       c3=-0.9421067708195129,    # c2+c4
                       c4=-1.177679984178871,
                       c5=0.13750633650504018,    # c4+c6
                       c6=1.3151863206839112):
        if update_tstep: p.update_tstep(p, self.eta)

        slow, fast = self.split(abs(tau), p)

        if slow.n:
            slow.set_dt_next(tau)
            if stream:
                stream.append(slow.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        if fast.n: fast = self.dkd67hcc(fast, c0 * tau / 2, False, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c0 * tau)
        if fast.n: fast = self.dkd67hcc(fast, c1 * tau / 2, True, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c2 * tau)
        if fast.n: fast = self.dkd67hcc(fast, c3 * tau / 2, True, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c4 * tau)
        if fast.n: fast = self.dkd67hcc(fast, c5 * tau / 2, True, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c6 * tau)
        if fast.n: fast = self.dkd67hcc(fast, c5 * tau / 2, True, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c4 * tau)
        if fast.n: fast = self.dkd67hcc(fast, c3 * tau / 2, True, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c2 * tau)
        if fast.n: fast = self.dkd67hcc(fast, c1 * tau / 2, True, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c0 * tau)
        if fast.n: fast = self.dkd67hcc(fast, c0 * tau / 2, True, stream)

        if slow.n:
            slow.set_dt_prev(tau)
            slow.update_t_curr(tau)
            slow.update_nstep()

        p = self.join(slow, fast)

        return p


    #
    # dkd69[std,shr,hcc] methods
    #

    def dkd69(self, p, tau, update_tstep, stream,
                    c0=0.3921614440073141,
                    c1=0.7247605807966735,      # c0+c2
                    c2=0.33259913678935943,
                    c3=-0.3736470357682799,     # c2+c4
                    c4=-0.7062461725576393,
                    c5=-0.6240325762640886,     # c4+c6
                    c6=0.0822135962935508,
                    c7=0.8807575872283808,      # c6+c8
                    c8=0.79854399093483):
        if update_tstep:
            p.update_tstep(p, self.eta)
            tau = self.get_min_block_tstep(p, tau)

        p.set_dt_next(tau)

        if stream:
            stream.append(p.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        p = self.dkd(p, c0 * tau)
        p = self.dkd(p, c2 * tau)
        p = self.dkd(p, c4 * tau)
        p = self.dkd(p, c6 * tau)
        p = self.dkd(p, c8 * tau)
        p = self.dkd(p, c6 * tau)
        p = self.dkd(p, c4 * tau)
        p = self.dkd(p, c2 * tau)
        p = self.dkd(p, c0 * tau)

        p.set_dt_prev(tau)
        p.update_t_curr(tau)
        p.update_nstep()

        return p


    def dkd69hcc(self, p, tau, update_tstep, stream,
                       c0=0.3921614440073141,
                       c1=0.7247605807966735,      # c0+c2
                       c2=0.33259913678935943,
                       c3=-0.3736470357682799,     # c2+c4
                       c4=-0.7062461725576393,
                       c5=-0.6240325762640886,     # c4+c6
                       c6=0.0822135962935508,
                       c7=0.8807575872283808,      # c6+c8
                       c8=0.79854399093483):
        if update_tstep: p.update_tstep(p, self.eta)

        slow, fast = self.split(abs(tau), p)

        if slow.n:
            slow.set_dt_next(tau)
            if stream:
                stream.append(slow.select(lambda x: x%self.dump_freq == 0, 'nstep'))

        if fast.n: fast = self.dkd67hcc(fast, c0 * tau / 2, False, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c0 * tau)
        if fast.n: fast = self.dkd67hcc(fast, c1 * tau / 2, True, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c2 * tau)
        if fast.n: fast = self.dkd67hcc(fast, c3 * tau / 2, True, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c4 * tau)
        if fast.n: fast = self.dkd67hcc(fast, c5 * tau / 2, True, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c6 * tau)
        if fast.n: fast = self.dkd67hcc(fast, c7 * tau / 2, True, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c8 * tau)
        if fast.n: fast = self.dkd67hcc(fast, c7 * tau / 2, True, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c6 * tau)
        if fast.n: fast = self.dkd67hcc(fast, c5 * tau / 2, True, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c4 * tau)
        if fast.n: fast = self.dkd67hcc(fast, c3 * tau / 2, True, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c2 * tau)
        if fast.n: fast = self.dkd67hcc(fast, c1 * tau / 2, True, stream)
        if slow.n: slow, fast = self.sf_dkd(slow, fast, c0 * tau)
        if fast.n: fast = self.dkd67hcc(fast, c0 * tau / 2, True, stream)

        if slow.n:
            slow.set_dt_prev(tau)
            slow.update_t_curr(tau)
            slow.update_nstep()

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

        if slow.n > 0: slow, fast = self.sf_dkd(slow, fast, tau / 2)
        if fast.n > 0: fast = self.meth1(fast, tau / 2, True, stream)
        if fast.n > 0: fast = self.meth1(fast, tau / 2, True, stream)
        if slow.n > 0: slow, fast = self.sf_dkd(slow, fast, tau / 2)

        if slow.n:
            slow.set_dt_prev(tau)
            slow.update_t_curr(tau)
            slow.update_nstep()

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
        if slow.n > 0: slow, empty = self.sf_dkd(slow, p.empty, tau)
        if fast.n > 0: fast = self.meth2(fast, tau / 2, True, stream)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau / 2)

        if slow.n:
            slow.set_dt_prev(tau)
            slow.update_t_curr(tau)
            slow.update_nstep()

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
        if slow.n > 0: slow, empty = self.sf_dkd(slow, p.empty, tau)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau / 2)
        if fast.n > 0: fast = self.meth3(fast, tau / 2, True, stream)

        if slow.n:
            slow.set_dt_prev(tau)
            slow.update_t_curr(tau)
            slow.update_nstep()

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

        if slow.n > 0: slow, empty = self.sf_dkd(slow, p.empty, tau / 2)
        if fast.n > 0: fast = self.meth4(fast, tau / 2, False, stream)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau)
        if fast.n > 0: fast = self.meth4(fast, tau / 2, True, stream)
        if slow.n > 0: slow, empty = self.sf_dkd(slow, p.empty, tau / 2)

        if slow.n:
            slow.set_dt_prev(tau)
            slow.update_t_curr(tau)
            slow.update_nstep()

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
        if slow.n > 0: slow, empty = self.sf_dkd(slow, p.empty, tau / 2)
        if slow.n > 0 and fast.n > 0: self.kick(slow, fast, tau)
        if slow.n > 0 and fast.n > 0: self.kick(fast, slow, tau)
        if slow.n > 0: slow, empty = self.sf_dkd(slow, p.empty, tau / 2)
        if fast.n > 0: fast = self.meth5(fast, tau / 2, True, stream)

        if slow.n:
            slow.set_dt_prev(tau)
            slow.update_t_curr(tau)
            slow.update_nstep()

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

        if slow.n:
            slow.set_dt_prev(tau)
            slow.update_t_curr(tau)
            slow.update_nstep()

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

        if slow.n:
            slow.set_dt_prev(tau)
            slow.update_t_curr(tau)
            slow.update_nstep()

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

        if slow.n:
            slow.set_dt_prev(tau)
            slow.update_t_curr(tau)
            slow.update_nstep()

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

        if slow.n:
            slow.set_dt_prev(tau)
            slow.update_t_curr(tau)
            slow.update_nstep()

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
