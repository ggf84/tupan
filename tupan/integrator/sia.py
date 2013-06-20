# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import logging
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
    for iobj in ips.members.values():
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
    for iobj in ips.members.values():
        (iobj.ax, iobj.ay, iobj.az) = iobj.get_acc(jps)
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

    def initialize(self, t_end):
        """

        """
        logger.info("Initializing '%s' integrator.",
                    self.method)

        ps = self.ps

        if self.reporter:
            self.reporter.diagnostic_report(self.time, ps)
        if self.dumpper:
            self.dumpper.dump_worldline(ps)

        self.is_initialized = True

    def finalize(self, t_end):
        """

        """
        logger.info("Finalizing '%s' integrator.",
                    self.method)

    def do_step(self, ps, tau):
        """

        """
        if "std" in self.method:
            self.update_tstep = False
            self.shared_tstep = True
        elif "shr" in self.method:
            self.update_tstep = True
            self.shared_tstep = True
        elif "hcc" in self.method:
            self.update_tstep = True
            self.shared_tstep = False
        else:
            raise ValueError("Unexpected method: {0}".format(self.method))

        if "dkd21" in self.method:
            return self.dkd21(ps, tau)
        elif "dkd22" in self.method:
            return self.dkd22(ps, tau)
        elif "dkd43" in self.method:
            return self.dkd43(ps, tau)
        elif "dkd44" in self.method:
            return self.dkd44(ps, tau)
        elif "dkd45" in self.method:
            return self.dkd45(ps, tau)
        elif "dkd46" in self.method:
            return self.dkd46(ps, tau)
        elif "dkd67" in self.method:
            return self.dkd67(ps, tau)
        elif "dkd69" in self.method:
            return self.dkd69(ps, tau)
        else:
            raise ValueError("Unexpected method: {0}".format(self.method))

    def recurse(self, ps, tau, sfdkdxy):
        """

        """
        flag = 0
        if self.update_tstep:
            flag = 1
            ps.update_tstep(ps, self.eta)
            if self.shared_tstep:
                ps.tstep = ps.min_tstep()
#                tau = self.get_min_block_tstep(ps, tau)

        slow, fast = split(ps, abs(ps.tstep) >= flag*abs(tau))

        slow, fast = sfdkdxy(slow, fast, tau, self.recurse)

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
    def sfdkd21(self, slow, fast, tau, recurse):
        """

        """
        if fast.n == 0:
            slow = base_dkd21(slow, tau)
            self.time += tau
            return slow, fast

        k, d = dkd21_coefs
        #
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd21)
        slow = base_dkd21(slow, d[0] * tau)
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd21)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = recurse(fast, d[0] * tau / 2, self.sfdkd21)
        slow = base_dkd21(slow, d[0] * tau)
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd21)
        #
        return slow, fast

    def dkd21(self, ps, tau):
        """

        """
        return self.recurse(ps, tau, self.sfdkd21)

    #
    # dkd22[std,shr,hcc] method -- D.K.D.K.D
    #
    def sfdkd22(self, slow, fast, tau, recurse):
        """

        """
        if fast.n == 0:
            slow = base_dkd22(slow, tau)
            self.time += tau
            return slow, fast

        k, d = dkd22_coefs
        #
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd22)
        slow = base_dkd22(slow, d[0] * tau)
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd22)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = recurse(fast, d[1] * tau / 2, self.sfdkd22)
        slow = base_dkd22(slow, d[1] * tau)
        fast = recurse(fast, d[1] * tau / 2, self.sfdkd22)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = recurse(fast, d[0] * tau / 2, self.sfdkd22)
        slow = base_dkd22(slow, d[0] * tau)
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd22)
        #
        return slow, fast

    def dkd22(self, ps, tau):
        """

        """
        return self.recurse(ps, tau, self.sfdkd22)

    #
    # dkd43[std,shr,hcc] method -- D.K.D.K.D.K.D
    #
    def sfdkd43(self, slow, fast, tau, recurse):
        """

        """
        if fast.n == 0:
            slow = base_dkd43(slow, tau)
            self.time += tau
            return slow, fast

        k, d = dkd43_coefs
        #
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd43)
        slow = base_dkd43(slow, d[0] * tau)
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd43)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = recurse(fast, d[1] * tau / 2, self.sfdkd43)
        slow = base_dkd43(slow, d[1] * tau)
        fast = recurse(fast, d[1] * tau / 2, self.sfdkd43)

        slow, fast = kick_sf(slow, fast, k[1] * tau)

        fast = recurse(fast, d[1] * tau / 2, self.sfdkd43)
        slow = base_dkd43(slow, d[1] * tau)
        fast = recurse(fast, d[1] * tau / 2, self.sfdkd43)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = recurse(fast, d[0] * tau / 2, self.sfdkd43)
        slow = base_dkd43(slow, d[0] * tau)
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd43)
        #
        return slow, fast

    def dkd43(self, ps, tau):
        """

        """
        return self.recurse(ps, tau, self.sfdkd43)

    #
    # dkd44[std,shr,hcc] method -- D.K.D.K.D.K.D.K.D
    #
    def sfdkd44(self, slow, fast, tau, recurse):
        """

        """
        if fast.n == 0:
            slow = base_dkd44(slow, tau)
            self.time += tau
            return slow, fast

        k, d = dkd44_coefs
        #
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd44)
        slow = base_dkd44(slow, d[0] * tau)
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd44)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = recurse(fast, d[1] * tau / 2, self.sfdkd44)
        slow = base_dkd44(slow, d[1] * tau)
        fast = recurse(fast, d[1] * tau / 2, self.sfdkd44)

        slow, fast = kick_sf(slow, fast, k[1] * tau)

        fast = recurse(fast, d[2] * tau / 2, self.sfdkd44)
        slow = base_dkd44(slow, d[2] * tau)
        fast = recurse(fast, d[2] * tau / 2, self.sfdkd44)

        slow, fast = kick_sf(slow, fast, k[1] * tau)

        fast = recurse(fast, d[1] * tau / 2, self.sfdkd44)
        slow = base_dkd44(slow, d[1] * tau)
        fast = recurse(fast, d[1] * tau / 2, self.sfdkd44)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = recurse(fast, d[0] * tau / 2, self.sfdkd44)
        slow = base_dkd44(slow, d[0] * tau)
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd44)
        #
        return slow, fast

    def dkd44(self, ps, tau):
        """

        """
        return self.recurse(ps, tau, self.sfdkd44)

    #
    # dkd45[std,shr,hcc] method -- D.K.D.K.D.K.D.K.D.K.D
    #
    def sfdkd45(self, slow, fast, tau, recurse):
        """

        """
        if fast.n == 0:
            slow = base_dkd45(slow, tau)
            self.time += tau
            return slow, fast

        k, d = dkd45_coefs
        #
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd45)
        slow = base_dkd45(slow, d[0] * tau)
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd45)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = recurse(fast, d[1] * tau / 2, self.sfdkd45)
        slow = base_dkd45(slow, d[1] * tau)
        fast = recurse(fast, d[1] * tau / 2, self.sfdkd45)

        slow, fast = kick_sf(slow, fast, k[1] * tau)

        fast = recurse(fast, d[2] * tau / 2, self.sfdkd45)
        slow = base_dkd45(slow, d[2] * tau)
        fast = recurse(fast, d[2] * tau / 2, self.sfdkd45)

        slow, fast = kick_sf(slow, fast, k[2] * tau)

        fast = recurse(fast, d[2] * tau / 2, self.sfdkd45)
        slow = base_dkd45(slow, d[2] * tau)
        fast = recurse(fast, d[2] * tau / 2, self.sfdkd45)

        slow, fast = kick_sf(slow, fast, k[1] * tau)

        fast = recurse(fast, d[1] * tau / 2, self.sfdkd45)
        slow = base_dkd45(slow, d[1] * tau)
        fast = recurse(fast, d[1] * tau / 2, self.sfdkd45)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = recurse(fast, d[0] * tau / 2, self.sfdkd45)
        slow = base_dkd45(slow, d[0] * tau)
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd45)
        #
        return slow, fast

    def dkd45(self, ps, tau):
        """

        """
        return self.recurse(ps, tau, self.sfdkd45)

    #
    # dkd46[std,shr,hcc] method -- D.K.D.K.D.K.D.K.D.K.D.K.D
    #
    def sfdkd46(self, slow, fast, tau, recurse):
        """

        """
        if fast.n == 0:
            slow = base_dkd46(slow, tau)
            self.time += tau
            return slow, fast

        k, d = dkd46_coefs
        #
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd46)
        slow = base_dkd46(slow, d[0] * tau)
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd46)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = recurse(fast, d[1] * tau / 2, self.sfdkd46)
        slow = base_dkd46(slow, d[1] * tau)
        fast = recurse(fast, d[1] * tau / 2, self.sfdkd46)

        slow, fast = kick_sf(slow, fast, k[1] * tau)

        fast = recurse(fast, d[2] * tau / 2, self.sfdkd46)
        slow = base_dkd46(slow, d[2] * tau)
        fast = recurse(fast, d[2] * tau / 2, self.sfdkd46)

        slow, fast = kick_sf(slow, fast, k[2] * tau)

        fast = recurse(fast, d[3] * tau / 2, self.sfdkd46)
        slow = base_dkd46(slow, d[3] * tau)
        fast = recurse(fast, d[3] * tau / 2, self.sfdkd46)

        slow, fast = kick_sf(slow, fast, k[2] * tau)

        fast = recurse(fast, d[2] * tau / 2, self.sfdkd46)
        slow = base_dkd46(slow, d[2] * tau)
        fast = recurse(fast, d[2] * tau / 2, self.sfdkd46)

        slow, fast = kick_sf(slow, fast, k[1] * tau)

        fast = recurse(fast, d[1] * tau / 2, self.sfdkd46)
        slow = base_dkd46(slow, d[1] * tau)
        fast = recurse(fast, d[1] * tau / 2, self.sfdkd46)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = recurse(fast, d[0] * tau / 2, self.sfdkd46)
        slow = base_dkd46(slow, d[0] * tau)
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd46)
        #
        return slow, fast

    def dkd46(self, ps, tau):
        """

        """
        return self.recurse(ps, tau, self.sfdkd46)

    #
    # dkd67[std,shr,hcc] method -- D.K.D.K.D.K.D.K.D.K.D.K.D.K.D
    #
    def sfdkd67(self, slow, fast, tau, recurse):
        """

        """
        if fast.n == 0:
            slow = base_dkd67(slow, tau)
            self.time += tau
            return slow, fast

        k, d = dkd67_coefs
        #
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd67)
        slow = base_dkd67(slow, d[0] * tau)
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd67)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = recurse(fast, d[1] * tau / 2, self.sfdkd67)
        slow = base_dkd67(slow, d[1] * tau)
        fast = recurse(fast, d[1] * tau / 2, self.sfdkd67)

        slow, fast = kick_sf(slow, fast, k[1] * tau)

        fast = recurse(fast, d[2] * tau / 2, self.sfdkd67)
        slow = base_dkd67(slow, d[2] * tau)
        fast = recurse(fast, d[2] * tau / 2, self.sfdkd67)

        slow, fast = kick_sf(slow, fast, k[2] * tau)

        fast = recurse(fast, d[3] * tau / 2, self.sfdkd67)
        slow = base_dkd67(slow, d[3] * tau)
        fast = recurse(fast, d[3] * tau / 2, self.sfdkd67)

        slow, fast = kick_sf(slow, fast, k[3] * tau)

        fast = recurse(fast, d[3] * tau / 2, self.sfdkd67)
        slow = base_dkd67(slow, d[3] * tau)
        fast = recurse(fast, d[3] * tau / 2, self.sfdkd67)

        slow, fast = kick_sf(slow, fast, k[2] * tau)

        fast = recurse(fast, d[2] * tau / 2, self.sfdkd67)
        slow = base_dkd67(slow, d[2] * tau)
        fast = recurse(fast, d[2] * tau / 2, self.sfdkd67)

        slow, fast = kick_sf(slow, fast, k[1] * tau)

        fast = recurse(fast, d[1] * tau / 2, self.sfdkd67)
        slow = base_dkd67(slow, d[1] * tau)
        fast = recurse(fast, d[1] * tau / 2, self.sfdkd67)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = recurse(fast, d[0] * tau / 2, self.sfdkd67)
        slow = base_dkd67(slow, d[0] * tau)
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd67)
        #
        return slow, fast

    def dkd67(self, ps, tau):
        """

        """
        return self.recurse(ps, tau, self.sfdkd67)

    #
    # dkd69[std,shr,hcc] method -- D.K.D.K.D.K.D.K.D.K.D.K.D.K.D.K.D.K.D
    #
    def sfdkd69(self, slow, fast, tau, recurse):
        """

        """
        if fast.n == 0:
            slow = base_dkd69(slow, tau)
            self.time += tau
            return slow, fast

        k, d = dkd69_coefs
        #
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd69)
        slow = base_dkd69(slow, d[0] * tau)
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd69)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = recurse(fast, d[1] * tau / 2, self.sfdkd69)
        slow = base_dkd69(slow, d[1] * tau)
        fast = recurse(fast, d[1] * tau / 2, self.sfdkd69)

        slow, fast = kick_sf(slow, fast, k[1] * tau)

        fast = recurse(fast, d[2] * tau / 2, self.sfdkd69)
        slow = base_dkd69(slow, d[2] * tau)
        fast = recurse(fast, d[2] * tau / 2, self.sfdkd69)

        slow, fast = kick_sf(slow, fast, k[2] * tau)

        fast = recurse(fast, d[3] * tau / 2, self.sfdkd69)
        slow = base_dkd69(slow, d[3] * tau)
        fast = recurse(fast, d[3] * tau / 2, self.sfdkd69)

        slow, fast = kick_sf(slow, fast, k[3] * tau)

        fast = recurse(fast, d[4] * tau / 2, self.sfdkd69)
        slow = base_dkd69(slow, d[4] * tau)
        fast = recurse(fast, d[4] * tau / 2, self.sfdkd69)

        slow, fast = kick_sf(slow, fast, k[4] * tau)

        fast = recurse(fast, d[4] * tau / 2, self.sfdkd69)
        slow = base_dkd69(slow, d[4] * tau)
        fast = recurse(fast, d[4] * tau / 2, self.sfdkd69)

        slow, fast = kick_sf(slow, fast, k[3] * tau)

        fast = recurse(fast, d[3] * tau / 2, self.sfdkd69)
        slow = base_dkd69(slow, d[3] * tau)
        fast = recurse(fast, d[3] * tau / 2, self.sfdkd69)

        slow, fast = kick_sf(slow, fast, k[2] * tau)

        fast = recurse(fast, d[2] * tau / 2, self.sfdkd69)
        slow = base_dkd69(slow, d[2] * tau)
        fast = recurse(fast, d[2] * tau / 2, self.sfdkd69)

        slow, fast = kick_sf(slow, fast, k[1] * tau)

        fast = recurse(fast, d[1] * tau / 2, self.sfdkd69)
        slow = base_dkd69(slow, d[1] * tau)
        fast = recurse(fast, d[1] * tau / 2, self.sfdkd69)

        slow, fast = kick_sf(slow, fast, k[0] * tau)

        fast = recurse(fast, d[0] * tau / 2, self.sfdkd69)
        slow = base_dkd69(slow, d[0] * tau)
        fast = recurse(fast, d[0] * tau / 2, self.sfdkd69)
        #
        return slow, fast

    def dkd69(self, ps, tau):
        """

        """
        return self.recurse(ps, tau, self.sfdkd69)


########## end of file ##########
