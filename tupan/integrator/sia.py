# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import logging
from ..integrator import Base
from ..lib.utils import ctype
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
    """Splits the particle's system into slow/fast components.

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
    """Joins the slow/fast components of a particle's system.

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
    """Drift operator for Newtonian quantities.

    """
    ips.rx += ips.vx * tau
    ips.ry += ips.vy * tau
    ips.rz += ips.vz * tau
    return ips


#
# kick_n
#
@timings
def kick_n(ips, jps, tau):
    """Kick operator for Newtonian quantities.

    """
    ips.set_acc(jps)
    ips.vx += ips.ax * tau
    ips.vy += ips.ay * tau
    ips.vz += ips.az * tau
    return ips


#
# drift_pn
#
@timings
def drift_pn(ips, tau):
    """Drift operator for post-Newtonian quantities.

    """
    ips = drift_n(ips, tau)
    ips.pn_drift_rcom(tau)
    return ips


#
# kick_pn
#
@timings
def kick_pn(ips, jps, tau):
    """Kick operator for post-Newtonian quantities.

    """
    ips.set_acc(jps)

    ips.set_pnacc(jps)
    ips.wx += (ips.ax + ips.pnax) * tau / 2
    ips.wy += (ips.ay + ips.pnay) * tau / 2
    ips.wz += (ips.az + ips.pnaz) * tau / 2

    ips.vx, ips.wx = ips.wx, ips.vx
    ips.vy, ips.wy = ips.wy, ips.vy
    ips.vz, ips.wz = ips.wz, ips.vz
    ips.set_pnacc(jps)
    ips.vx, ips.wx = ips.wx, ips.vx
    ips.vy, ips.wy = ips.wy, ips.vy
    ips.vz, ips.wz = ips.wz, ips.vz

    ips.pn_kick_ke(tau / 2)
    ips.pn_kick_lmom(tau / 2)
    ips.pn_kick_amom(tau / 2)

    ips.vx += (ips.ax + ips.pnax) * tau
    ips.vy += (ips.ay + ips.pnay) * tau
    ips.vz += (ips.az + ips.pnaz) * tau

    ips.pn_kick_ke(tau / 2)
    ips.pn_kick_lmom(tau / 2)
    ips.pn_kick_amom(tau / 2)

    ips.set_pnacc(jps)
    ips.wx += (ips.ax + ips.pnax) * tau / 2
    ips.wy += (ips.ay + ips.pnay) * tau / 2
    ips.wz += (ips.az + ips.pnaz) * tau / 2

    return ips


#
# drift
#
@timings
def drift(ips, tau):
    """Drift operator.

    """
    if INCLUDE_PN_CORRECTIONS:
        return drift_pn(ips, tau)
    return drift_n(ips, tau)


#
# kick
#
@timings
def kick(ips, jps, tau, pn):
    """Kick operator.

    """
    if pn and INCLUDE_PN_CORRECTIONS:
        return kick_pn(ips, jps, tau)
    return kick_n(ips, jps, tau)


#
# sf_drift
#
@timings
def sf_drift(slow, fast, tau, evolve, recurse, bridge):
    """Slow<->Fast Drift operator.

    """
    slow = evolve(slow, tau)
    fast = recurse(fast, tau, evolve, bridge)
    return slow, fast


#
# sf_kick
#
@timings
def sf_kick(slow, fast, tau):
    """Slow<->Fast Kick operator.

    """
    if slow.n and fast.n:
        slow = kick(slow, fast, tau, pn=True)
        fast = kick(fast, slow, tau, pn=True)
    return slow, fast


#
# SIA21
#
class SIA21(object):
    # REF.: Yoshida, Phys. Lett. A 150 (1990)
    coefs = ([1.0],
             [0.5])

    @staticmethod
    @timings
    def dkd(ips, tau):
        """Standard SIA21 DKD-type propagator.

        """
        if ips.n == 0:
            return ips

        if ips.n == 1:
            return drift(ips, tau)

        k, d = SIA21.coefs

        ips = drift(ips, d[0] * tau)
        ips = kick(ips, ips, k[0] * tau, pn=True)
        ips = drift(ips, d[0] * tau)

        return ips

    @staticmethod
    @timings
    def kdk(ips, tau):
        """Standard SIA21 KDK-type propagator.

        """
        if ips.n == 0:
            return ips

        if ips.n == 1:
            return drift(ips, tau)

        d, k = SIA21.coefs

        ips = kick(ips, ips, k[0] * tau, pn=True)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, ips, k[0] * tau, pn=True)

        return ips

    @staticmethod
    @timings
    def bridge_sf(slow, fast, tau, evolve, recurse):
        """

        """
        k, d = SIA21.coefs
        erb = evolve, recurse, SIA21.bridge_sf
        #
        slow, fast = sf_drift(slow, fast, d[0] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[0] * tau)
        slow, fast = sf_drift(slow, fast, d[0] * tau, *erb)
        #
        return slow, fast


#
# SIA22
#
class SIA22(object):
    # REF.: Omelyan, Mryglod & Folk, Comput. Phys. Comm. 151 (2003)
    coefs = ([0.5],
             [0.1931833275037836,
              0.6136333449924328])

    @staticmethod
    @timings
    def dkd(ips, tau):
        """Standard SIA22 DKD-type propagator.

        """
        if ips.n == 0:
            return ips

        if ips.n == 1:
            return drift(ips, tau)

        k, d = SIA22.coefs

        ips = drift(ips, d[0] * tau)
        ips = kick(ips, ips, k[0] * tau, pn=True)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, ips, k[0] * tau, pn=True)
        ips = drift(ips, d[0] * tau)

        return ips

    @staticmethod
    @timings
    def kdk(ips, tau):
        """Standard SIA22 KDK-type propagator.

        """
        if ips.n == 0:
            return ips

        if ips.n == 1:
            return drift(ips, tau)

        d, k = SIA22.coefs

        ips = kick(ips, ips, k[0] * tau, pn=True)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, ips, k[1] * tau, pn=True)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, ips, k[0] * tau, pn=True)

        return ips

    @staticmethod
    @timings
    def bridge_sf(slow, fast, tau, evolve, recurse):
        """

        """
        k, d = SIA22.coefs
        erb = evolve, recurse, SIA22.bridge_sf
        #
        slow, fast = sf_drift(slow, fast, d[0] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[0] * tau)
        slow, fast = sf_drift(slow, fast, d[1] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[0] * tau)
        slow, fast = sf_drift(slow, fast, d[0] * tau, *erb)
        #
        return slow, fast


#
# SIA43
#
class SIA43(object):
    # REF.: Yoshida, Phys. Lett. A 150 (1990)
    coefs = ([1.3512071919596575,
              -1.7024143839193150],
             [0.6756035959798288,
              -0.17560359597982877])

    @staticmethod
    @timings
    def dkd(ips, tau):
        """Standard SIA43 DKD-type propagator.

        """
        if ips.n == 0:
            return ips

        if ips.n == 1:
            return drift(ips, tau)

        k, d = SIA43.coefs

        ips = drift(ips, d[0] * tau)
        ips = kick(ips, ips, k[0] * tau, pn=True)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, ips, k[1] * tau, pn=True)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, ips, k[0] * tau, pn=True)
        ips = drift(ips, d[0] * tau)

        return ips

    @staticmethod
    @timings
    def kdk(ips, tau):
        """Standard SIA43 KDK-type propagator.

        """
        if ips.n == 0:
            return ips

        if ips.n == 1:
            return drift(ips, tau)

        d, k = SIA43.coefs

        ips = kick(ips, ips, k[0] * tau, pn=True)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, ips, k[1] * tau, pn=True)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, ips, k[1] * tau, pn=True)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, ips, k[0] * tau, pn=True)

        return ips

    @staticmethod
    @timings
    def bridge_sf(slow, fast, tau, evolve, recurse):
        """

        """
        k, d = SIA43.coefs
        erb = evolve, recurse, SIA43.bridge_sf
        #
        slow, fast = sf_drift(slow, fast, d[0] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[0] * tau)
        slow, fast = sf_drift(slow, fast, d[1] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[1] * tau)
        slow, fast = sf_drift(slow, fast, d[1] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[0] * tau)
        slow, fast = sf_drift(slow, fast, d[0] * tau, *erb)
        #
        return slow, fast


#
# SIA44
#
class SIA44(object):
    # REF.: Omelyan, Mryglod & Folk, Comput. Phys. Comm. 151 (2003)
    coefs = ([0.7123418310626056,
              -0.21234183106260562],
             [0.1786178958448091,
              -0.06626458266981843,
              0.7752933736500186])

    @staticmethod
    @timings
    def dkd(ips, tau):
        """Standard SIA44 DKD-type propagator.

        """
        if ips.n == 0:
            return ips

        if ips.n == 1:
            return drift(ips, tau)

        k, d = SIA44.coefs

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

    @staticmethod
    @timings
    def kdk(ips, tau):
        """Standard SIA44 KDK-type propagator.

        """
        if ips.n == 0:
            return ips

        if ips.n == 1:
            return drift(ips, tau)

        d, k = SIA44.coefs

        ips = kick(ips, ips, k[0] * tau, pn=True)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, ips, k[1] * tau, pn=True)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, ips, k[2] * tau, pn=True)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, ips, k[1] * tau, pn=True)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, ips, k[0] * tau, pn=True)

        return ips

    @staticmethod
    @timings
    def bridge_sf(slow, fast, tau, evolve, recurse):
        """

        """
        k, d = SIA44.coefs
        erb = evolve, recurse, SIA44.bridge_sf
        #
        slow, fast = sf_drift(slow, fast, d[0] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[0] * tau)
        slow, fast = sf_drift(slow, fast, d[1] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[1] * tau)
        slow, fast = sf_drift(slow, fast, d[2] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[1] * tau)
        slow, fast = sf_drift(slow, fast, d[1] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[0] * tau)
        slow, fast = sf_drift(slow, fast, d[0] * tau, *erb)
        #
        return slow, fast


#
# SIA45
#
class SIA45(object):
    # REF.: Omelyan, Mryglod & Folk, Comput. Phys. Comm. 151 (2003)
    coefs = ([-0.0844296195070715,
              0.354900057157426,
              0.459059124699291],
             [0.2750081212332419,
              -0.1347950099106792,
              0.35978688867743724])

    @staticmethod
    @timings
    def dkd(ips, tau):
        """Standard SIA45 DKD-type propagator.

        """
        if ips.n == 0:
            return ips

        if ips.n == 1:
            return drift(ips, tau)

        k, d = SIA45.coefs

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

    @staticmethod
    @timings
    def kdk(ips, tau):
        """Standard SIA45 KDK-type propagator.

        """
        if ips.n == 0:
            return ips

        if ips.n == 1:
            return drift(ips, tau)

        d, k = SIA45.coefs

        ips = kick(ips, ips, k[0] * tau, pn=True)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, ips, k[1] * tau, pn=True)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, ips, k[2] * tau, pn=True)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, ips, k[2] * tau, pn=True)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, ips, k[1] * tau, pn=True)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, ips, k[0] * tau, pn=True)

        return ips

    @staticmethod
    @timings
    def bridge_sf(slow, fast, tau, evolve, recurse):
        """

        """
        k, d = SIA45.coefs
        erb = evolve, recurse, SIA45.bridge_sf
        #
        slow, fast = sf_drift(slow, fast, d[0] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[0] * tau)
        slow, fast = sf_drift(slow, fast, d[1] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[1] * tau)
        slow, fast = sf_drift(slow, fast, d[2] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[2] * tau)
        slow, fast = sf_drift(slow, fast, d[2] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[1] * tau)
        slow, fast = sf_drift(slow, fast, d[1] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[0] * tau)
        slow, fast = sf_drift(slow, fast, d[0] * tau, *erb)
        #
        return slow, fast


#
# SIA46
#
class SIA46(object):
    # REF.: Blanes & Moan, J. Comp. Appl. Math. 142 (2002)
    coefs = ([0.209515106613362,
              -0.143851773179818,
              0.434336666566456],
             [0.0792036964311957,
              0.353172906049774,
              -0.0420650803577195,
              0.21937695575349958])

    @staticmethod
    @timings
    def dkd(ips, tau):
        """Standard SIA46 DKD-type propagator.

        """
        if ips.n == 0:
            return ips

        if ips.n == 1:
            return drift(ips, tau)

        k, d = SIA46.coefs

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

    @staticmethod
    @timings
    def kdk(ips, tau):
        """Standard SIA46 KDK-type propagator.

        """
        if ips.n == 0:
            return ips

        if ips.n == 1:
            return drift(ips, tau)

        d, k = SIA46.coefs

        ips = kick(ips, ips, k[0] * tau, pn=True)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, ips, k[1] * tau, pn=True)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, ips, k[2] * tau, pn=True)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, ips, k[3] * tau, pn=True)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, ips, k[2] * tau, pn=True)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, ips, k[1] * tau, pn=True)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, ips, k[0] * tau, pn=True)

        return ips

    @staticmethod
    @timings
    def bridge_sf(slow, fast, tau, evolve, recurse):
        """

        """
        k, d = SIA46.coefs
        erb = evolve, recurse, SIA46.bridge_sf
        #
        slow, fast = sf_drift(slow, fast, d[0] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[0] * tau)
        slow, fast = sf_drift(slow, fast, d[1] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[1] * tau)
        slow, fast = sf_drift(slow, fast, d[2] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[2] * tau)
        slow, fast = sf_drift(slow, fast, d[3] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[2] * tau)
        slow, fast = sf_drift(slow, fast, d[2] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[1] * tau)
        slow, fast = sf_drift(slow, fast, d[1] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[0] * tau)
        slow, fast = sf_drift(slow, fast, d[0] * tau, *erb)
        #
        return slow, fast


#
# SIA67
#
class SIA67(object):
    # REF.: Yoshida, Phys. Lett. A 150 (1990)
    coefs = ([0.7845136104775573,
              0.23557321335935813,
              -1.177679984178871,
              1.3151863206839112],
             [0.39225680523877865,
              0.5100434119184577,
              -0.47105338540975644,
              0.06875316825252015])

    @staticmethod
    @timings
    def dkd(ips, tau):
        """Standard SIA67 DKD-type propagator.

        """
        if ips.n == 0:
            return ips

        if ips.n == 1:
            return drift(ips, tau)

        k, d = SIA67.coefs

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

    @staticmethod
    @timings
    def kdk(ips, tau):
        """Standard SIA67 KDK-type propagator.

        """
        if ips.n == 0:
            return ips

        if ips.n == 1:
            return drift(ips, tau)

        d, k = SIA67.coefs

        ips = kick(ips, ips, k[0] * tau, pn=True)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, ips, k[1] * tau, pn=True)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, ips, k[2] * tau, pn=True)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, ips, k[3] * tau, pn=True)
        ips = drift(ips, d[3] * tau)
        ips = kick(ips, ips, k[3] * tau, pn=True)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, ips, k[2] * tau, pn=True)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, ips, k[1] * tau, pn=True)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, ips, k[0] * tau, pn=True)

        return ips

    @staticmethod
    @timings
    def bridge_sf(slow, fast, tau, evolve, recurse):
        """

        """
        k, d = SIA67.coefs
        erb = evolve, recurse, SIA67.bridge_sf
        #
        slow, fast = sf_drift(slow, fast, d[0] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[0] * tau)
        slow, fast = sf_drift(slow, fast, d[1] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[1] * tau)
        slow, fast = sf_drift(slow, fast, d[2] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[2] * tau)
        slow, fast = sf_drift(slow, fast, d[3] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[3] * tau)
        slow, fast = sf_drift(slow, fast, d[3] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[2] * tau)
        slow, fast = sf_drift(slow, fast, d[2] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[1] * tau)
        slow, fast = sf_drift(slow, fast, d[1] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[0] * tau)
        slow, fast = sf_drift(slow, fast, d[0] * tau, *erb)
        #
        return slow, fast


#
# SIA69
#
class SIA69(object):
    # REF.: Kahan & Li, Math. Comput. 66 (1997)
    coefs = ([0.39103020330868477,
              0.334037289611136,
              -0.7062272811875614,
              0.08187754964805945,
              0.7985644772393624],
             [0.19551510165434238,
              0.3625337464599104,
              -0.1860949957882127,
              -0.31217486576975095,
              0.44022101344371095])

    @staticmethod
    @timings
    def dkd(ips, tau):
        """Standard SIA69 DKD-type propagator.

        """
        if ips.n == 0:
            return ips

        if ips.n == 1:
            return drift(ips, tau)

        k, d = SIA69.coefs

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

    @staticmethod
    @timings
    def kdk(ips, tau):
        """Standard SIA69 KDK-type propagator.

        """
        if ips.n == 0:
            return ips

        if ips.n == 1:
            return drift(ips, tau)

        d, k = SIA69.coefs

        ips = kick(ips, ips, k[0] * tau, pn=True)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, ips, k[1] * tau, pn=True)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, ips, k[2] * tau, pn=True)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, ips, k[3] * tau, pn=True)
        ips = drift(ips, d[3] * tau)
        ips = kick(ips, ips, k[4] * tau, pn=True)
        ips = drift(ips, d[4] * tau)
        ips = kick(ips, ips, k[4] * tau, pn=True)
        ips = drift(ips, d[3] * tau)
        ips = kick(ips, ips, k[3] * tau, pn=True)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, ips, k[2] * tau, pn=True)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, ips, k[1] * tau, pn=True)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, ips, k[0] * tau, pn=True)

        return ips

    @staticmethod
    @timings
    def bridge_sf(slow, fast, tau, evolve, recurse):
        """

        """
        k, d = SIA69.coefs
        erb = evolve, recurse, SIA69.bridge_sf
        #
        slow, fast = sf_drift(slow, fast, d[0] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[0] * tau)
        slow, fast = sf_drift(slow, fast, d[1] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[1] * tau)
        slow, fast = sf_drift(slow, fast, d[2] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[2] * tau)
        slow, fast = sf_drift(slow, fast, d[3] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[3] * tau)
        slow, fast = sf_drift(slow, fast, d[4] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[4] * tau)
        slow, fast = sf_drift(slow, fast, d[4] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[3] * tau)
        slow, fast = sf_drift(slow, fast, d[3] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[2] * tau)
        slow, fast = sf_drift(slow, fast, d[2] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[1] * tau)
        slow, fast = sf_drift(slow, fast, d[1] * tau, *erb)
        slow, fast = sf_kick(slow, fast, k[0] * tau)
        slow, fast = sf_drift(slow, fast, d[0] * tau, *erb)
        #
        return slow, fast


@decallmethods(timings)
class SIA(Base):
    """

    """
    PROVIDED_METHODS = ['sia21s.dkd', 'sia21a.dkd', 'sia21h.dkd',
                        'sia21s.kdk', 'sia21a.kdk', 'sia21h.kdk',
                        'sia22s.dkd', 'sia22a.dkd', 'sia22h.dkd',
                        'sia22s.kdk', 'sia22a.kdk', 'sia22h.kdk',
                        'sia43s.dkd', 'sia43a.dkd', 'sia43h.dkd',
                        'sia43s.kdk', 'sia43a.kdk', 'sia43h.kdk',
                        'sia44s.dkd', 'sia44a.dkd', 'sia44h.dkd',
                        'sia44s.kdk', 'sia44a.kdk', 'sia44h.kdk',
                        'sia45s.dkd', 'sia45a.dkd', 'sia45h.dkd',
                        'sia45s.kdk', 'sia45a.kdk', 'sia45h.kdk',
                        'sia46s.dkd', 'sia46a.dkd', 'sia46h.dkd',
                        'sia46s.kdk', 'sia46a.kdk', 'sia46h.kdk',
                        'sia67s.dkd', 'sia67a.dkd', 'sia67h.dkd',
                        'sia67s.kdk', 'sia67a.kdk', 'sia67h.kdk',
                        'sia69s.dkd', 'sia69a.dkd', 'sia69h.dkd',
                        'sia69s.kdk', 'sia69a.kdk', 'sia69h.kdk',
                        ]

    def __init__(self, eta, time, ps, method, **kwargs):
        """

        """
        super(SIA, self).__init__(eta, time, ps, **kwargs)
        self.method = method
        global INCLUDE_PN_CORRECTIONS
        INCLUDE_PN_CORRECTIONS = self.include_pn_corrections

    def initialize(self, t_end):
        """

        """
        logger.info("Initializing '%s' integrator.",
                    self.method)

        ps = self.ps
        if INCLUDE_PN_CORRECTIONS:
            ps.register_auxiliary_attribute("wx", ctype.REAL)
            ps.register_auxiliary_attribute("wy", ctype.REAL)
            ps.register_auxiliary_attribute("wz", ctype.REAL)
            ps.wx[:] = ps.vx
            ps.wy[:] = ps.vy
            ps.wz[:] = ps.vz

        if self.reporter:
            self.reporter.diagnostic_report(ps)
        if self.dumpper:
            self.dumpper.dump_worldline(ps)
        if self.viewer:
            self.viewer.show_event(ps)

        self.is_initialized = True

    def finalize(self, t_end):
        """

        """
        logger.info("Finalizing '%s' integrator.",
                    self.method)

        ps = self.ps

        if self.viewer:
            self.viewer.show_event(ps)
            self.viewer.enter_main_loop()

    def do_step(self, ps, tau):
        """

        """
        if "s." in self.method:
            self.update_tstep = False
            self.shared_tstep = True
        elif "a." in self.method:
            self.update_tstep = True
            self.shared_tstep = True
        elif "h." in self.method:
            self.update_tstep = True
            self.shared_tstep = False
        else:
            raise ValueError("Unexpected method: {0}".format(self.method))

        if "dkd" in self.method:
            if "sia21" in self.method:
                return self.recurse(ps, tau, SIA21.dkd, SIA21.bridge_sf)
            elif "sia22" in self.method:
                return self.recurse(ps, tau, SIA22.dkd, SIA22.bridge_sf)
            elif "sia43" in self.method:
                return self.recurse(ps, tau, SIA43.dkd, SIA43.bridge_sf)
            elif "sia44" in self.method:
                return self.recurse(ps, tau, SIA44.dkd, SIA44.bridge_sf)
            elif "sia45" in self.method:
                return self.recurse(ps, tau, SIA45.dkd, SIA45.bridge_sf)
            elif "sia46" in self.method:
                return self.recurse(ps, tau, SIA46.dkd, SIA46.bridge_sf)
            elif "sia67" in self.method:
                return self.recurse(ps, tau, SIA67.dkd, SIA67.bridge_sf)
            elif "sia69" in self.method:
                return self.recurse(ps, tau, SIA69.dkd, SIA69.bridge_sf)
            else:
                raise ValueError("Unexpected method: {0}".format(self.method))
        elif "kdk" in self.method:
            if "sia21" in self.method:
                return self.recurse(ps, tau, SIA21.kdk, SIA21.bridge_sf)
            elif "sia22" in self.method:
                return self.recurse(ps, tau, SIA22.kdk, SIA22.bridge_sf)
            elif "sia43" in self.method:
                return self.recurse(ps, tau, SIA43.kdk, SIA43.bridge_sf)
            elif "sia44" in self.method:
                return self.recurse(ps, tau, SIA44.kdk, SIA44.bridge_sf)
            elif "sia45" in self.method:
                return self.recurse(ps, tau, SIA45.kdk, SIA45.bridge_sf)
            elif "sia46" in self.method:
                return self.recurse(ps, tau, SIA46.kdk, SIA46.bridge_sf)
            elif "sia67" in self.method:
                return self.recurse(ps, tau, SIA67.kdk, SIA67.bridge_sf)
            elif "sia69" in self.method:
                return self.recurse(ps, tau, SIA69.kdk, SIA69.bridge_sf)
            else:
                raise ValueError("Unexpected method: {0}".format(self.method))
        else:
            raise ValueError("Unexpected method: {0}".format(self.method))

    def recurse(self, ps, tau, evolve, bridge_sf):
        """

        """
        if not ps.n:
            return ps

        flag = -1
        if self.update_tstep:
            flag = 1
            ps.set_tstep(ps, self.eta)
            if self.shared_tstep:
                tau = self.get_min_block_tstep(ps, tau)

        slow, fast = split(ps, abs(ps.tstep) > flag*abs(tau))

        slow, fast = bridge_sf(slow, fast, tau, evolve, self.recurse)

        if fast.n == 0:
            type(ps).t_curr += tau

        if slow.n:
            slow.tstep[:] = tau
            slow.time += tau
            slow.nstep += 1
            if self.dumpper:
                slc = slow.time % (self.dump_freq * tau) == 0
                if any(slc):
                    self.wl.append(slow[slc])
            if self.viewer:
                slc = slow.time % (self.gl_freq * tau) == 0
                if any(slc):
                    self.viewer.show_event(slow[slc])

        return join(slow, fast)


########## end of file ##########
