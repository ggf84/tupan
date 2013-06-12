# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import math
import logging
import numpy as np
from ..integrator import Base
from ..lib import gravity
from .sakura import sakura_step
from .nreg import nreg_step
from ..lib.utils.timing import decallmethods, timings


__all__ = ["SIA"]

logger = logging.getLogger(__name__)

#
# global variables
#
INCLUDE_PN_CORRECTIONS = False


#
# split
#
@timings
def split(ps, condition):
    """
    Splits the particle's system into slow/fast components.
    """
    slow = ps[condition]
    fast = ps[~condition]

    if slow.n + fast.n != ps.n:
        logger.error(
            "slow.n + fast.n != ps.n: %d, %d, %d.",
            slow.n, fast.n, ps.n)

    return slow, fast


#
# join
#
@timings
def join(slow, fast):
    """
    Joins the slow/fast components of a particle's system.
    """
    if not fast.n:
        return slow
    if not slow.n:
        return fast
    ps = slow.copy()
    ps.append(fast)
    return ps


#
# drift_n
#
@timings
def drift_n(ips, tau):
    """
    Drift operator for Newtonian quantities.
    """
    for iobj in ips.members:
        iobj.rx += tau * iobj.vx
        iobj.ry += tau * iobj.vy
        iobj.rz += tau * iobj.vz
    return ips


#
# kick_n
#
@timings
def kick_n(ips, jps, tau):
    """
    Kick operator for Newtonian quantities.
    """
    (ips.ax, ips.ay, ips.az) = ips.get_acc(jps)
    for iobj in ips.members:
        iobj.vx += tau * iobj.ax
        iobj.vy += tau * iobj.ay
        iobj.vz += tau * iobj.az
    return ips


#
# drift_pn
#
@timings
def drift_pn(ips, tau):
    """
    Drift operator for post-Newtonian quantities.
    """
    ips = drift_n(ips, tau)
    ips.evolve_rcom_pn_shift(tau)
    return ips


#
# kick_pn
#
@timings
def kick_pn(ips, jps, tau):
    """
    Kick operator for post-Newtonian quantities.
    """
    ips = kick_n(ips, jps, tau / 2)

    ips.vx += ips.pn_dvx
    ips.vy += ips.pn_dvy
    ips.vz += ips.pn_dvz

    ips.evolve_ke_pn_shift(tau / 2)
    ips.evolve_lmom_pn_shift(tau / 2)
    ips.evolve_amom_pn_shift(tau / 2)

    (pnax, pnay, pnaz) = ips.get_pnacc(jps)
    g = 2 * tau * 0
    ips.pn_dvx = (tau * pnax - (1 - g) * ips.pn_dvx) / (1 + g)
    ips.pn_dvy = (tau * pnay - (1 - g) * ips.pn_dvy) / (1 + g)
    ips.pn_dvz = (tau * pnaz - (1 - g) * ips.pn_dvz) / (1 + g)

    ips.evolve_amom_pn_shift(tau / 2)
    ips.evolve_lmom_pn_shift(tau / 2)
    ips.evolve_ke_pn_shift(tau / 2)

    ips.vz += ips.pn_dvz
    ips.vy += ips.pn_dvy
    ips.vx += ips.pn_dvx

    ips = kick_n(ips, jps, tau / 2)
    return ips


#
# drift
#
@timings
def drift(ips, tau):
    """
    Drift operator.
    """
    if INCLUDE_PN_CORRECTIONS:
        return drift_pn(ips, tau)
    return drift_n(ips, tau)


#
# kick
#
@timings
def kick(ips, jps, tau, pn):
    """
    Kick operator.
    """
    if pn and INCLUDE_PN_CORRECTIONS:
        return kick_pn(ips, jps, tau)
    return kick_n(ips, jps, tau)


#
# kick_sf
#
@timings
def kick_sf(slow, fast, tau):
    """
    Slow<->Fast Kick operator.
    """
    if slow.n and fast.n:
        slow = kick(slow, fast, tau, pn=True)
        fast = kick(fast, slow, tau, pn=True)
    return slow, fast


#
# dkd21
#
dkd21_coefs = ([1.0],
               [0.5])


@timings
def base_dkd21(ips, tau):
    """
    Standard dkd21 operator.
    """
    if ips.n == 0:
        return ips

    if ips.n == 1:
        return drift(ips, tau)

    k, d = dkd21_coefs

    ips = drift(ips, d[0] * tau)
    ips = kick(ips, ips, k[0] * tau, pn=True)
    ips = drift(ips, d[0] * tau)

    return ips


#
# dkd22
#
dkd22_coefs = ([0.5],
               [0.1931833275037836,
                0.6136333449924328])


@timings
def base_dkd22(ips, tau):
    """
    Standard dkd22 operator.
    """
    if ips.n == 0:
        return ips

    if ips.n == 1:
        return drift(ips, tau)

    k, d = dkd22_coefs

    ips = drift(ips, d[0] * tau)
    ips = kick(ips, ips, k[0] * tau, pn=True)
    ips = drift(ips, d[1] * tau)
    ips = kick(ips, ips, k[0] * tau, pn=True)
    ips = drift(ips, d[0] * tau)

    return ips


#
# dkd43
#
dkd43_coefs = ([1.3512071919596575,
                -1.7024143839193150],
               [0.6756035959798288,
                -0.17560359597982877])


@timings
def base_dkd43(ips, tau):
    """
    Standard dkd43 operator.
    """
    if ips.n == 0:
        return ips

    if ips.n == 1:
        return drift(ips, tau)

    k, d = dkd43_coefs

    ips = drift(ips, d[0] * tau)
    ips = kick(ips, ips, k[0] * tau, pn=True)
    ips = drift(ips, d[1] * tau)
    ips = kick(ips, ips, k[1] * tau, pn=True)
    ips = drift(ips, d[1] * tau)
    ips = kick(ips, ips, k[0] * tau, pn=True)
    ips = drift(ips, d[0] * tau)

    return ips


#
# dkd44
#
dkd44_coefs = ([0.7123418310626056,
                -0.21234183106260562],
               [0.1786178958448091,
                -0.06626458266981843,
                0.7752933736500186])


@timings
def base_dkd44(ips, tau):
    """
    Standard dkd44 operator.
    """
    if ips.n == 0:
        return ips

    if ips.n == 1:
        return drift(ips, tau)

    k, d = dkd44_coefs

    ips = drift(ips, d[0] * tau)
    ips = kick(ips, ips, k[0] * tau, pn=True)
    ips = drift(ips, d[1] * tau)
    ips = kick(ips, ips, k[1] * tau, pn=True)
    ips = drift(ips, d[2] * tau)
    ips = kick(ips, ips, k[1] * tau, pn=True)
    ips = drift(ips, d[1] * tau)
    ips = kick(ips, ips, k[0] * tau, pn=True)
    ips = drift(ips, d[0] * tau)

    return ips


#
# dkd45
#
dkd45_coefs = ([-0.0844296195070715,
                0.354900057157426,
                0.459059124699291],
               [0.2750081212332419,
                -0.1347950099106792,
                0.35978688867743724])


@timings
def base_dkd45(ips, tau):
    """
    Standard dkd45 operator.
    """
    if ips.n == 0:
        return ips

    if ips.n == 1:
        return drift(ips, tau)

    k, d = dkd45_coefs

    ips = drift(ips, d[0] * tau)
    ips = kick(ips, ips, k[0] * tau, pn=True)
    ips = drift(ips, d[1] * tau)
    ips = kick(ips, ips, k[1] * tau, pn=True)
    ips = drift(ips, d[2] * tau)
    ips = kick(ips, ips, k[2] * tau, pn=True)
    ips = drift(ips, d[2] * tau)
    ips = kick(ips, ips, k[1] * tau, pn=True)
    ips = drift(ips, d[1] * tau)
    ips = kick(ips, ips, k[0] * tau, pn=True)
    ips = drift(ips, d[0] * tau)

    return ips


#
# dkd46
#
dkd46_coefs = ([0.209515106613362,
                -0.143851773179818,
                0.434336666566456],
               [0.0792036964311957,
                0.353172906049774,
                -0.0420650803577195,
                0.21937695575349958])


@timings
def base_dkd46(ips, tau):
    """
    Standard dkd46 operator.
    """
    if ips.n == 0:
        return ips

    if ips.n == 1:
        return drift(ips, tau)

    k, d = dkd46_coefs

    ips = drift(ips, d[0] * tau)
    ips = kick(ips, ips, k[0] * tau, pn=True)
    ips = drift(ips, d[1] * tau)
    ips = kick(ips, ips, k[1] * tau, pn=True)
    ips = drift(ips, d[2] * tau)
    ips = kick(ips, ips, k[2] * tau, pn=True)
    ips = drift(ips, d[3] * tau)
    ips = kick(ips, ips, k[2] * tau, pn=True)
    ips = drift(ips, d[2] * tau)
    ips = kick(ips, ips, k[1] * tau, pn=True)
    ips = drift(ips, d[1] * tau)
    ips = kick(ips, ips, k[0] * tau, pn=True)
    ips = drift(ips, d[0] * tau)

    return ips


#
# dkd67
#
dkd67_coefs = ([0.7845136104775573,
                0.23557321335935813,
                -1.177679984178871,
                1.3151863206839112],
               [0.39225680523877865,
                0.5100434119184577,
                -0.47105338540975644,
                0.06875316825252015])


@timings
def base_dkd67(ips, tau):
    """
    Standard dkd67 operator.
    """
    if ips.n == 0:
        return ips

    if ips.n == 1:
        return drift(ips, tau)

    k, d = dkd67_coefs

    ips = drift(ips, d[0] * tau)
    ips = kick(ips, ips, k[0] * tau, pn=True)
    ips = drift(ips, d[1] * tau)
    ips = kick(ips, ips, k[1] * tau, pn=True)
    ips = drift(ips, d[2] * tau)
    ips = kick(ips, ips, k[2] * tau, pn=True)
    ips = drift(ips, d[3] * tau)
    ips = kick(ips, ips, k[3] * tau, pn=True)
    ips = drift(ips, d[3] * tau)
    ips = kick(ips, ips, k[2] * tau, pn=True)
    ips = drift(ips, d[2] * tau)
    ips = kick(ips, ips, k[1] * tau, pn=True)
    ips = drift(ips, d[1] * tau)
    ips = kick(ips, ips, k[0] * tau, pn=True)
    ips = drift(ips, d[0] * tau)

    return ips


#
# dkd69
#
dkd69_coefs = ([0.39103020330868477,
                0.334037289611136,
                -0.7062272811875614,
                0.08187754964805945,
                0.7985644772393624],
               [0.19551510165434238,
                0.3625337464599104,
                -0.1860949957882127,
                -0.31217486576975095,
                0.44022101344371095])


@timings
def base_dkd69(ips, tau):
    """
    Standard dkd69 operator.
    """
    if ips.n == 0:
        return ips

    if ips.n == 1:
        return drift(ips, tau)

    k, d = dkd69_coefs

    ips = drift(ips, d[0] * tau)
    ips = kick(ips, ips, k[0] * tau, pn=True)
    ips = drift(ips, d[1] * tau)
    ips = kick(ips, ips, k[1] * tau, pn=True)
    ips = drift(ips, d[2] * tau)
    ips = kick(ips, ips, k[2] * tau, pn=True)
    ips = drift(ips, d[3] * tau)
    ips = kick(ips, ips, k[3] * tau, pn=True)
    ips = drift(ips, d[4] * tau)
    ips = kick(ips, ips, k[4] * tau, pn=True)
    ips = drift(ips, d[4] * tau)
    ips = kick(ips, ips, k[3] * tau, pn=True)
    ips = drift(ips, d[3] * tau)
    ips = kick(ips, ips, k[2] * tau, pn=True)
    ips = drift(ips, d[2] * tau)
    ips = kick(ips, ips, k[1] * tau, pn=True)
    ips = drift(ips, d[1] * tau)
    ips = kick(ips, ips, k[0] * tau, pn=True)
    ips = drift(ips, d[0] * tau)

    return ips


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

    def __init__(self, eta, time, ps, method, **kwargs):
        """

        """
        super(SIA, self).__init__(eta, time, ps, **kwargs)
        self.method = method
        global INCLUDE_PN_CORRECTIONS
        INCLUDE_PN_CORRECTIONS = True if gravity.clight.pn_order > 0 else False

    def get_base_tstep(self, t_end):
        """

        """
        self.tstep = 0.125
        if abs(self.time + self.tstep) > t_end:
            self.tstep = math.copysign(t_end - abs(self.time), self.eta)
        return self.tstep

    def initialize(self, t_end):
        """

        """
        logger.info(
            "Initializing '%s' integrator.",
            self.method
        )

        ps = self.ps
        (ps.ax, ps.ay, ps.az) = ps.get_acc(ps)

        if self.reporter:
            self.reporter.diagnostic_report(self.time, ps)
        if self.dumpper:
            self.dumpper.dump_worldline(ps)

        self.is_initialized = True

    def finalize(self, t_end):
        """

        """
        logger.info(
            "Finalizing '%s' integrator.",
            self.method
        )

    def get_min_block_tstep(self, ps, tau):
        """

        """
        min_tstep = ps.min_tstep()

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

    def do_step(self, ps, tau):
        """

        """
        if self.method == "sia.dkd21std":
            return self.dkd21(ps, tau, False, True)
        elif self.method == "sia.dkd21shr":
            return self.dkd21(ps, tau, True, True)
        elif self.method == "sia.dkd21hcc":
            return self.dkd21(ps, tau, True, False)
        elif self.method == "sia.dkd22std":
            return self.dkd22(ps, tau, False, True)
        elif self.method == "sia.dkd22shr":
            return self.dkd22(ps, tau, True, True)
        elif self.method == "sia.dkd22hcc":
            return self.dkd22(ps, tau, True, False)
        elif self.method == "sia.dkd43std":
            return self.dkd43(ps, tau, False, True)
        elif self.method == "sia.dkd43shr":
            return self.dkd43(ps, tau, True, True)
        elif self.method == "sia.dkd43hcc":
            return self.dkd43(ps, tau, True, False)
        elif self.method == "sia.dkd44std":
            return self.dkd44(ps, tau, False, True)
        elif self.method == "sia.dkd44shr":
            return self.dkd44(ps, tau, True, True)
        elif self.method == "sia.dkd44hcc":
            return self.dkd44(ps, tau, True, False)
        elif self.method == "sia.dkd45std":
            return self.dkd45(ps, tau, False, True)
        elif self.method == "sia.dkd45shr":
            return self.dkd45(ps, tau, True, True)
        elif self.method == "sia.dkd45hcc":
            return self.dkd45(ps, tau, True, False)
        elif self.method == "sia.dkd46std":
            return self.dkd46(ps, tau, False, True)
        elif self.method == "sia.dkd46shr":
            return self.dkd46(ps, tau, True, True)
        elif self.method == "sia.dkd46hcc":
            return self.dkd46(ps, tau, True, False)
        elif self.method == "sia.dkd67std":
            return self.dkd67(ps, tau, False, True)
        elif self.method == "sia.dkd67shr":
            return self.dkd67(ps, tau, True, True)
        elif self.method == "sia.dkd67hcc":
            return self.dkd67(ps, tau, True, False)
        elif self.method == "sia.dkd69std":
            return self.dkd69(ps, tau, False, True)
        elif self.method == "sia.dkd69shr":
            return self.dkd69(ps, tau, True, True)
        elif self.method == "sia.dkd69hcc":
            return self.dkd69(ps, tau, True, False)
        else:
            raise ValueError("Unexpected method: {0}".format(self.method))

    def evolve_step(self, t_end):
        """

        """
        if not self.is_initialized:
            self.initialize(t_end)

        ps = self.ps
        tau = self.get_base_tstep(t_end)
        self.wl = ps[:0]

        ps = self.do_step(ps, tau)

        self.time += self.tstep
        self.ps = ps

        if self.reporter:
            self.reporter.diagnostic_report(self.time, ps)
        if self.dumpper:
            self.dumpper.dump_worldline(self.wl)

    def rdkdxy(self, ps, tau, sfdkdxy, update_tstep, shared_tstep=False):
        """

        """
        flag = 0
        if update_tstep:
            flag = 1
            ps.update_tstep(ps, self.eta)
            if shared_tstep:
                tau = self.get_min_block_tstep(ps, tau)

        slow, fast = split(ps, abs(ps.tstep) >= flag*abs(tau))

        slow, fast = sfdkdxy(slow, fast, tau, sfdkdxy)

        if slow.n:
            slow.tstep = tau
            slow.time += tau
            slow.nstep += 1
            wp = slow[slow.time % (self.dump_freq * tau) == 0]
            if wp.n:
                self.wl.append(wp.copy())

        return join(slow, fast)

    #
    # dkd21[std,shr,hcc] method -- D.K.D
    #
    def dkd21_sf(self, slow, fast, tau, sfdkdxy):
        """

        """
        if fast.n == 0:
            slow = base_dkd21(slow, tau)
            return slow, fast

        k, d = dkd21_coefs
        #
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        slow = base_dkd21(slow, d[0] * tau)
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        slow = base_dkd21(slow, d[0] * tau)
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        #
        return slow, fast

    def dkd21(self, ps, tau, update_tstep, shared_tstep=False):
        """

        """
        return self.rdkdxy(ps, tau, self.dkd21_sf, update_tstep, shared_tstep)

    #
    # dkd22[std,shr,hcc] method -- D.K.D.K.D
    #
    def dkd22_sf(self, slow, fast, tau, sfdkdxy):
        """

        """
        if fast.n == 0:
            slow = base_dkd22(slow, tau)
            return slow, fast

        k, d = dkd22_coefs
        #
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        slow = base_dkd22(slow, d[0] * tau)
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)
        slow = base_dkd22(slow, d[1] * tau)
        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        slow = base_dkd22(slow, d[0] * tau)
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        #
        return slow, fast

    def dkd22(self, ps, tau, update_tstep, shared_tstep=False):
        """

        """
        return self.rdkdxy(ps, tau, self.dkd22_sf, update_tstep, shared_tstep)

    #
    # dkd43[std,shr,hcc] method -- D.K.D.K.D.K.D
    #
    def dkd43_sf(self, slow, fast, tau, sfdkdxy):
        """

        """
        if fast.n == 0:
            slow = base_dkd43(slow, tau)
            return slow, fast

        k, d = dkd43_coefs
        #
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        slow = base_dkd43(slow, d[0] * tau)
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)
        slow = base_dkd43(slow, d[1] * tau)
        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[1] * tau)

        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)
        slow = base_dkd43(slow, d[1] * tau)
        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        slow = base_dkd43(slow, d[0] * tau)
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        #
        return slow, fast

    def dkd43(self, ps, tau, update_tstep, shared_tstep=False):
        """

        """
        return self.rdkdxy(ps, tau, self.dkd43_sf, update_tstep, shared_tstep)

    #
    # dkd44[std,shr,hcc] method -- D.K.D.K.D.K.D.K.D
    #
    def dkd44_sf(self, slow, fast, tau, sfdkdxy):
        """

        """
        if fast.n == 0:
            slow = base_dkd44(slow, tau)
            return slow, fast

        k, d = dkd44_coefs
        #
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        slow = base_dkd44(slow, d[0] * tau)
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)
        slow = base_dkd44(slow, d[1] * tau)
        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[1] * tau)

        fast = self.rdkdxy(fast, d[2] * tau / 2, sfdkdxy, True)
        slow = base_dkd44(slow, d[2] * tau)
        fast = self.rdkdxy(fast, d[2] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[1] * tau)

        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)
        slow = base_dkd44(slow, d[1] * tau)
        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        slow = base_dkd44(slow, d[0] * tau)
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        #
        return slow, fast

    def dkd44(self, ps, tau, update_tstep, shared_tstep=False):
        """

        """
        return self.rdkdxy(ps, tau, self.dkd44_sf, update_tstep, shared_tstep)

    #
    # dkd45[std,shr,hcc] method -- D.K.D.K.D.K.D.K.D.K.D
    #
    def dkd45_sf(self, slow, fast, tau, sfdkdxy):
        """

        """
        if fast.n == 0:
            slow = base_dkd45(slow, tau)
            return slow, fast

        k, d = dkd45_coefs
        #
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        slow = base_dkd45(slow, d[0] * tau)
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)
        slow = base_dkd45(slow, d[1] * tau)
        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[1] * tau)

        fast = self.rdkdxy(fast, d[2] * tau / 2, sfdkdxy, True)
        slow = base_dkd45(slow, d[2] * tau)
        fast = self.rdkdxy(fast, d[2] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[2] * tau)

        fast = self.rdkdxy(fast, d[2] * tau / 2, sfdkdxy, True)
        slow = base_dkd45(slow, d[2] * tau)
        fast = self.rdkdxy(fast, d[2] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[1] * tau)

        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)
        slow = base_dkd45(slow, d[1] * tau)
        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        slow = base_dkd45(slow, d[0] * tau)
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        #
        return slow, fast

    def dkd45(self, ps, tau, update_tstep, shared_tstep=False):
        """

        """
        return self.rdkdxy(ps, tau, self.dkd45_sf, update_tstep, shared_tstep)

    #
    # dkd46[std,shr,hcc] method -- D.K.D.K.D.K.D.K.D.K.D.K.D
    #
    def dkd46_sf(self, slow, fast, tau, sfdkdxy):
        """

        """
        if fast.n == 0:
            slow = base_dkd46(slow, tau)
            return slow, fast

        k, d = dkd46_coefs
        #
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        slow = base_dkd46(slow, d[0] * tau)
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)
        slow = base_dkd46(slow, d[1] * tau)
        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[1] * tau)

        fast = self.rdkdxy(fast, d[2] * tau / 2, sfdkdxy, True)
        slow = base_dkd46(slow, d[2] * tau)
        fast = self.rdkdxy(fast, d[2] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[2] * tau)

        fast = self.rdkdxy(fast, d[3] * tau / 2, sfdkdxy, True)
        slow = base_dkd46(slow, d[3] * tau)
        fast = self.rdkdxy(fast, d[3] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[2] * tau)

        fast = self.rdkdxy(fast, d[2] * tau / 2, sfdkdxy, True)
        slow = base_dkd46(slow, d[2] * tau)
        fast = self.rdkdxy(fast, d[2] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[1] * tau)

        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)
        slow = base_dkd46(slow, d[1] * tau)
        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        slow = base_dkd46(slow, d[0] * tau)
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        #
        return slow, fast

    def dkd46(self, ps, tau, update_tstep, shared_tstep=False):
        """

        """
        return self.rdkdxy(ps, tau, self.dkd46_sf, update_tstep, shared_tstep)

    #
    # dkd67[std,shr,hcc] method -- D.K.D.K.D.K.D.K.D.K.D.K.D.K.D
    #
    def dkd67_sf(self, slow, fast, tau, sfdkdxy):
        """

        """
        if fast.n == 0:
            slow = base_dkd67(slow, tau)
            return slow, fast

        k, d = dkd67_coefs
        #
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        slow = base_dkd67(slow, d[0] * tau)
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)
        slow = base_dkd67(slow, d[1] * tau)
        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[1] * tau)

        fast = self.rdkdxy(fast, d[2] * tau / 2, sfdkdxy, True)
        slow = base_dkd67(slow, d[2] * tau)
        fast = self.rdkdxy(fast, d[2] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[2] * tau)

        fast = self.rdkdxy(fast, d[3] * tau / 2, sfdkdxy, True)
        slow = base_dkd67(slow, d[3] * tau)
        fast = self.rdkdxy(fast, d[3] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[3] * tau)

        fast = self.rdkdxy(fast, d[3] * tau / 2, sfdkdxy, True)
        slow = base_dkd67(slow, d[3] * tau)
        fast = self.rdkdxy(fast, d[3] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[2] * tau)

        fast = self.rdkdxy(fast, d[2] * tau / 2, sfdkdxy, True)
        slow = base_dkd67(slow, d[2] * tau)
        fast = self.rdkdxy(fast, d[2] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[1] * tau)

        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)
        slow = base_dkd67(slow, d[1] * tau)
        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        slow = base_dkd67(slow, d[0] * tau)
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        #
        return slow, fast

    def dkd67(self, ps, tau, update_tstep, shared_tstep=False):
        """

        """
        return self.rdkdxy(ps, tau, self.dkd67_sf, update_tstep, shared_tstep)

    #
    # dkd69[std,shr,hcc] method -- D.K.D.K.D.K.D.K.D.K.D.K.D.K.D.K.D.K.D
    #
    def dkd69_sf(self, slow, fast, tau, sfdkdxy):
        """

        """
        if fast.n == 0:
            slow = base_dkd69(slow, tau)
            return slow, fast

        k, d = dkd69_coefs
        #
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        slow = base_dkd69(slow, d[0] * tau)
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)
        slow = base_dkd69(slow, d[1] * tau)
        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[1] * tau)

        fast = self.rdkdxy(fast, d[2] * tau / 2, sfdkdxy, True)
        slow = base_dkd69(slow, d[2] * tau)
        fast = self.rdkdxy(fast, d[2] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[2] * tau)

        fast = self.rdkdxy(fast, d[3] * tau / 2, sfdkdxy, True)
        slow = base_dkd69(slow, d[3] * tau)
        fast = self.rdkdxy(fast, d[3] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[3] * tau)

        fast = self.rdkdxy(fast, d[4] * tau / 2, sfdkdxy, True)
        slow = base_dkd69(slow, d[4] * tau)
        fast = self.rdkdxy(fast, d[4] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[4] * tau)

        fast = self.rdkdxy(fast, d[4] * tau / 2, sfdkdxy, True)
        slow = base_dkd69(slow, d[4] * tau)
        fast = self.rdkdxy(fast, d[4] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[3] * tau)

        fast = self.rdkdxy(fast, d[3] * tau / 2, sfdkdxy, True)
        slow = base_dkd69(slow, d[3] * tau)
        fast = self.rdkdxy(fast, d[3] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[2] * tau)

        fast = self.rdkdxy(fast, d[2] * tau / 2, sfdkdxy, True)
        slow = base_dkd69(slow, d[2] * tau)
        fast = self.rdkdxy(fast, d[2] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[1] * tau)

        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)
        slow = base_dkd69(slow, d[1] * tau)
        fast = self.rdkdxy(fast, d[1] * tau / 2, sfdkdxy, True)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        slow = base_dkd69(slow, d[0] * tau)
        fast = self.rdkdxy(fast, d[0] * tau / 2, sfdkdxy, True)
        #
        return slow, fast

    def dkd69(self, ps, tau, update_tstep, shared_tstep=False):
        """

        """
        return self.rdkdxy(ps, tau, self.dkd69_sf, update_tstep, shared_tstep)


########## end of file ##########
