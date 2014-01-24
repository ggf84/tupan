# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import logging
from ..integrator import Base
from .fewbody import FewBody
from ..lib.utils.timing import decallmethods, timings


__all__ = ["SIA"]

logger = logging.getLogger(__name__)


#
# split
#
@timings
def split(ps, condition):
    """Splits the particle's system into slow/fast components.

    """
    if ps.n <= 2:       # stop recursion and use a few-body solver!
        return ps, type(ps)()

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
    ps = slow
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
def kick_n(ips, tau):
    """Kick operator for Newtonian quantities.

    """
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
    ips.rx += ips.vx * tau
    ips.ry += ips.vy * tau
    ips.rz += ips.vz * tau
    ips.pn_drift_com_r(tau)
    return ips


#
# kick_pn
#
@timings
def kick_pn(ips, tau):
    """Kick operator for post-Newtonian quantities.

    """
#    #
#    def vw_swap(ps):
#        ps.vx, ps.wx = ps.wx, ps.vx
#        ps.vy, ps.wy = ps.wy, ps.vy
#        ps.vz, ps.wz = ps.wz, ps.vz
#        return ps
#    ips.set_pnacc(ips)
#    ips.wx += (ips.ax + ips.pnax) * tau / 2
#    ips.wy += (ips.ay + ips.pnay) * tau / 2
#    ips.wz += (ips.az + ips.pnaz) * tau / 2
#
#    ips = vw_swap(ips)
#    ips.set_pnacc(ips)
#    ips.pn_kick_ke(tau)
#    ips.pn_kick_lmom(tau)
#    ips.pn_kick_amom(tau)
#    ips = vw_swap(ips)
#
#    ips.vx += (ips.ax + ips.pnax) * tau
#    ips.vy += (ips.ay + ips.pnay) * tau
#    ips.vz += (ips.az + ips.pnaz) * tau
#
#    ips.set_pnacc(ips)
#    ips.wx += (ips.ax + ips.pnax) * tau / 2
#    ips.wy += (ips.ay + ips.pnay) * tau / 2
#    ips.wz += (ips.az + ips.pnaz) * tau / 2
#    #

#####

    #
    ips.vx += (ips.ax * tau + ips.wx) / 2
    ips.vy += (ips.ay * tau + ips.wy) / 2
    ips.vz += (ips.az * tau + ips.wz) / 2

    ips.set_pnacc(ips)
    ips.pn_kick_ke(tau)
    ips.pn_kick_lmom(tau)
    ips.pn_kick_amom(tau)
    ips.wx[...] = 2 * ips.pnax * tau - ips.wx
    ips.wy[...] = 2 * ips.pnay * tau - ips.wy
    ips.wz[...] = 2 * ips.pnaz * tau - ips.wz

    ips.vx += (ips.ax * tau + ips.wx) / 2
    ips.vy += (ips.ay * tau + ips.wy) / 2
    ips.vz += (ips.az * tau + ips.wz) / 2
    #

    return ips


#
# drift
#
@timings
def drift(ips, tau):
    """Drift operator.

    """
    if ips.include_pn_corrections:
        return drift_pn(ips, tau)
    return drift_n(ips, tau)


#
# kick
#
@timings
def kick(ips, tau):
    """Kick operator.

    """
    ips.set_acc(ips)
    if ips.include_pn_corrections:
        return kick_pn(ips, tau)
    return kick_n(ips, tau)


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
        slow.set_acc(fast)
        fast.set_acc(slow)
        if slow.include_pn_corrections:
#            #
#            def vw_swap(ps):
#                ps.vx, ps.wx = ps.wx, ps.vx
#                ps.vy, ps.wy = ps.wy, ps.vy
#                ps.vz, ps.wz = ps.wz, ps.vz
#                return ps
#            slow.set_pnacc(fast)
#            fast.set_pnacc(slow)
#            slow.wx += (slow.ax + slow.pnax) * tau / 2
#            slow.wy += (slow.ay + slow.pnay) * tau / 2
#            slow.wz += (slow.az + slow.pnaz) * tau / 2
#            fast.wx += (fast.ax + fast.pnax) * tau / 2
#            fast.wy += (fast.ay + fast.pnay) * tau / 2
#            fast.wz += (fast.az + fast.pnaz) * tau / 2
#
#            slow = vw_swap(slow)
#            fast = vw_swap(fast)
#            slow.set_pnacc(fast)
#            fast.set_pnacc(slow)
#            slow.pn_kick_ke(tau)
#            slow.pn_kick_lmom(tau)
#            slow.pn_kick_amom(tau)
#            fast.pn_kick_ke(tau)
#            fast.pn_kick_lmom(tau)
#            fast.pn_kick_amom(tau)
#            slow = vw_swap(slow)
#            fast = vw_swap(fast)
#
#            slow.vx += (slow.ax + slow.pnax) * tau
#            slow.vy += (slow.ay + slow.pnay) * tau
#            slow.vz += (slow.az + slow.pnaz) * tau
#            fast.vx += (fast.ax + fast.pnax) * tau
#            fast.vy += (fast.ay + fast.pnay) * tau
#            fast.vz += (fast.az + fast.pnaz) * tau
#
#            slow.set_pnacc(fast)
#            fast.set_pnacc(slow)
#            slow.wx += (slow.ax + slow.pnax) * tau / 2
#            slow.wy += (slow.ay + slow.pnay) * tau / 2
#            slow.wz += (slow.az + slow.pnaz) * tau / 2
#            fast.wx += (fast.ax + fast.pnax) * tau / 2
#            fast.wy += (fast.ay + fast.pnay) * tau / 2
#            fast.wz += (fast.az + fast.pnaz) * tau / 2
#            #

#############

            #
            slow.vx += (slow.ax * tau + slow.wx) / 2
            slow.vy += (slow.ay * tau + slow.wy) / 2
            slow.vz += (slow.az * tau + slow.wz) / 2
            fast.vx += (fast.ax * tau + fast.wx) / 2
            fast.vy += (fast.ay * tau + fast.wy) / 2
            fast.vz += (fast.az * tau + fast.wz) / 2

            slow.set_pnacc(fast)
            fast.set_pnacc(slow)
            slow.pn_kick_ke(tau)
            slow.pn_kick_lmom(tau)
            slow.pn_kick_amom(tau)
            fast.pn_kick_ke(tau)
            fast.pn_kick_lmom(tau)
            fast.pn_kick_amom(tau)
            slow.wx[...] = 2 * slow.pnax * tau - slow.wx
            slow.wy[...] = 2 * slow.pnay * tau - slow.wy
            slow.wz[...] = 2 * slow.pnaz * tau - slow.wz
            fast.wx[...] = 2 * fast.pnax * tau - fast.wx
            fast.wy[...] = 2 * fast.pnay * tau - fast.wy
            fast.wz[...] = 2 * fast.pnaz * tau - fast.wz

            slow.vx += (slow.ax * tau + slow.wx) / 2
            slow.vy += (slow.ay * tau + slow.wy) / 2
            slow.vz += (slow.az * tau + slow.wz) / 2
            fast.vx += (fast.ax * tau + fast.wx) / 2
            fast.vy += (fast.ay * tau + fast.wy) / 2
            fast.vz += (fast.az * tau + fast.wz) / 2
            #
        else:
            slow = kick_n(slow, tau)
            fast = kick_n(fast, tau)
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
        if ips.n <= 2:
            return FewBody.evolve(ips, tau)

        k, d = SIA21.coefs

        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[0] * tau)

        return ips

    @staticmethod
    @timings
    def kdk(ips, tau):
        """Standard SIA21 KDK-type propagator.

        """
        if ips.n <= 2:
            return FewBody.evolve(ips, tau)

        d, k = SIA21.coefs

        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[0] * tau)

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
        if ips.n <= 2:
            return FewBody.evolve(ips, tau)

        k, d = SIA22.coefs

        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[0] * tau)

        return ips

    @staticmethod
    @timings
    def kdk(ips, tau):
        """Standard SIA22 KDK-type propagator.

        """
        if ips.n <= 2:
            return FewBody.evolve(ips, tau)

        d, k = SIA22.coefs

        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[0] * tau)

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
        if ips.n <= 2:
            return FewBody.evolve(ips, tau)

        k, d = SIA43.coefs

        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[0] * tau)

        return ips

    @staticmethod
    @timings
    def kdk(ips, tau):
        """Standard SIA43 KDK-type propagator.

        """
        if ips.n <= 2:
            return FewBody.evolve(ips, tau)

        d, k = SIA43.coefs

        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[0] * tau)

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
        if ips.n <= 2:
            return FewBody.evolve(ips, tau)

        k, d = SIA44.coefs

        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[0] * tau)

        return ips

    @staticmethod
    @timings
    def kdk(ips, tau):
        """Standard SIA44 KDK-type propagator.

        """
        if ips.n <= 2:
            return FewBody.evolve(ips, tau)

        d, k = SIA44.coefs

        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[2] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[0] * tau)

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
        if ips.n <= 2:
            return FewBody.evolve(ips, tau)

        k, d = SIA45.coefs

        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, k[2] * tau)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[0] * tau)

        return ips

    @staticmethod
    @timings
    def kdk(ips, tau):
        """Standard SIA45 KDK-type propagator.

        """
        if ips.n <= 2:
            return FewBody.evolve(ips, tau)

        d, k = SIA45.coefs

        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[2] * tau)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, k[2] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[0] * tau)

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
        if ips.n <= 2:
            return FewBody.evolve(ips, tau)

        k, d = SIA46.coefs

        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, k[2] * tau)
        ips = drift(ips, d[3] * tau)
        ips = kick(ips, k[2] * tau)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[0] * tau)

        return ips

    @staticmethod
    @timings
    def kdk(ips, tau):
        """Standard SIA46 KDK-type propagator.

        """
        if ips.n <= 2:
            return FewBody.evolve(ips, tau)

        d, k = SIA46.coefs

        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[2] * tau)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, k[3] * tau)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, k[2] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[0] * tau)

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
        if ips.n <= 2:
            return FewBody.evolve(ips, tau)

        k, d = SIA67.coefs

        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, k[2] * tau)
        ips = drift(ips, d[3] * tau)
        ips = kick(ips, k[3] * tau)
        ips = drift(ips, d[3] * tau)
        ips = kick(ips, k[2] * tau)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[0] * tau)

        return ips

    @staticmethod
    @timings
    def kdk(ips, tau):
        """Standard SIA67 KDK-type propagator.

        """
        if ips.n <= 2:
            return FewBody.evolve(ips, tau)

        d, k = SIA67.coefs

        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[2] * tau)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, k[3] * tau)
        ips = drift(ips, d[3] * tau)
        ips = kick(ips, k[3] * tau)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, k[2] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[0] * tau)

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
        if ips.n <= 2:
            return FewBody.evolve(ips, tau)

        k, d = SIA69.coefs

        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, k[2] * tau)
        ips = drift(ips, d[3] * tau)
        ips = kick(ips, k[3] * tau)
        ips = drift(ips, d[4] * tau)
        ips = kick(ips, k[4] * tau)
        ips = drift(ips, d[4] * tau)
        ips = kick(ips, k[3] * tau)
        ips = drift(ips, d[3] * tau)
        ips = kick(ips, k[2] * tau)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[0] * tau)

        return ips

    @staticmethod
    @timings
    def kdk(ips, tau):
        """Standard SIA69 KDK-type propagator.

        """
        if ips.n <= 2:
            return FewBody.evolve(ips, tau)

        d, k = SIA69.coefs

        ips = kick(ips, k[0] * tau)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[2] * tau)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, k[3] * tau)
        ips = drift(ips, d[3] * tau)
        ips = kick(ips, k[4] * tau)
        ips = drift(ips, d[4] * tau)
        ips = kick(ips, k[4] * tau)
        ips = drift(ips, d[3] * tau)
        ips = kick(ips, k[3] * tau)
        ips = drift(ips, d[2] * tau)
        ips = kick(ips, k[2] * tau)
        ips = drift(ips, d[1] * tau)
        ips = kick(ips, k[1] * tau)
        ips = drift(ips, d[0] * tau)
        ips = kick(ips, k[0] * tau)

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

    def initialize(self, t_end):
        """

        """
        logger.info("Initializing '%s' integrator.",
                    self.method)

        ps = self.ps
        if ps.include_pn_corrections:
            ps.register_auxiliary_attribute("wx", "real")
            ps.register_auxiliary_attribute("wy", "real")
            ps.register_auxiliary_attribute("wz", "real")
#            ps.wx[...] = ps.vx
#            ps.wy[...] = ps.vy
#            ps.wz[...] = ps.vz

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
            slow.tstep[...] = tau
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
