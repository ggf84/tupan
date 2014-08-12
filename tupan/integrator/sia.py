# -*- coding: utf-8 -*-
#

"""
TODO.
"""


import logging
from .base import Base
from ..lib import extensions as ext
from ..lib.utils.timing import timings, bind_all


__all__ = ['SIA']

LOGGER = logging.getLogger(__name__)


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
        LOGGER.error(
            'slow.n + fast.n != ps.n: %d, %d, %d.',
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
def drift_n(ips, dt):
    """Drift operator for Newtonian quantities.

    """
    ips.rx += ips.vx * dt
    ips.ry += ips.vy * dt
    ips.rz += ips.vz * dt
    return ips


#
# kick_n
#
@timings
def kick_n(ips, dt):
    """Kick operator for Newtonian quantities.

    """
    ips.vx += ips.ax * dt
    ips.vy += ips.ay * dt
    ips.vz += ips.az * dt
    return ips


#
# drift_pn
#
@timings
def drift_pn(ips, dt):
    """Drift operator for post-Newtonian quantities.

    """
    ips.rx += ips.vx * dt
    ips.ry += ips.vy * dt
    ips.rz += ips.vz * dt
    ips.pn_drift_com_r(dt)
    return ips


def vw_swap(ips):
    ips.vx, ips.wx = ips.wx, ips.vx
    ips.vy, ips.wy = ips.wy, ips.vy
    ips.vz, ips.wz = ips.wz, ips.vz
    return ips


#
# kick_pn
#
@timings
def kick_pn(ips, dt):
    """Kick operator for post-Newtonian quantities.

    """
    #
    ips.set_pnacc(ips)
    ips.pn_kick_ke(dt / 2)
    ips.pn_kick_lmom(dt / 2)
    ips.pn_kick_amom(dt / 2)

    ips.wx[...] = ips.vx
    ips.wy[...] = ips.vy
    ips.wz[...] = ips.vz
    ips.wx += (ips.ax + ips.pnax) * dt / 2
    ips.wy += (ips.ay + ips.pnay) * dt / 2
    ips.wz += (ips.az + ips.pnaz) * dt / 2

    ips = vw_swap(ips)
    ips.set_pnacc(ips)
    ips = vw_swap(ips)

    ips.vx += (ips.ax + ips.pnax) * dt
    ips.vy += (ips.ay + ips.pnay) * dt
    ips.vz += (ips.az + ips.pnaz) * dt

    ips.set_pnacc(ips)
    ips.wx += (ips.ax + ips.pnax) * dt / 2
    ips.wy += (ips.ay + ips.pnay) * dt / 2
    ips.wz += (ips.az + ips.pnaz) * dt / 2
    ips.vx[...] = ips.wx
    ips.vy[...] = ips.wy
    ips.vz[...] = ips.wz

    ips.pn_kick_ke(dt / 2)
    ips.pn_kick_lmom(dt / 2)
    ips.pn_kick_amom(dt / 2)
    #

    #
    # ips.vx += (ips.ax * dt + ips.wx) / 2
    # ips.vy += (ips.ay * dt + ips.wy) / 2
    # ips.vz += (ips.az * dt + ips.wz) / 2

    # ips.set_pnacc(ips)
    # ips.pn_kick_ke(dt)
    # ips.pn_kick_lmom(dt)
    # ips.pn_kick_amom(dt)
    # ips.wx[...] = 2 * ips.pnax * dt - ips.wx
    # ips.wy[...] = 2 * ips.pnay * dt - ips.wy
    # ips.wz[...] = 2 * ips.pnaz * dt - ips.wz

    # ips.vx += (ips.ax * dt + ips.wx) / 2
    # ips.vy += (ips.ay * dt + ips.wy) / 2
    # ips.vz += (ips.az * dt + ips.wz) / 2
    #

    return ips


#
# drift
#
@timings
def drift(ips, dt):
    """Drift operator.

    """
    if hasattr(ips, 'include_pn_corrections'):
        return drift_pn(ips, dt)
    return drift_n(ips, dt)


#
# kick
#
@timings
def kick(ips, dt):
    """Kick operator.

    """
    ips.set_acc(ips)
    if hasattr(ips, 'include_pn_corrections'):
        return kick_pn(ips, dt)
    return kick_n(ips, dt)


#
# twobody_solver
#
@timings
def twobody_solver(ips, dt, kernel=ext.Kepler()):
    """

    """
    if hasattr(ips, 'include_pn_corrections'):
        raise NotImplementedError('The current version of the '
                                  'Kepler-solver does not include '
                                  'post-Newtonian corrections.')
    else:
        kernel(ips, ips, dt=dt)
    return ips


#
# fewbody_solver
#
@timings
def fewbody_solver(ips, dt):
    """

    """
    if ips.n == 1:
        return drift(ips, dt)
    return twobody_solver(ips, dt)


#
# sf_drift
#
@timings
def sf_drift(slow, fast, dt, evolve, recurse):
    """Slow<->Fast Drift operator.

    """
    slow = evolve(slow, dt) if slow.n else slow
    fast = recurse(fast, dt) if fast.n else fast
    return slow, fast


#
# sf_kick
#
@timings
def sf_kick(slow, fast, dt):
    """Slow<->Fast Kick operator.

    """
    if not (slow.n and fast.n):
        return slow, fast

    slow.set_acc(fast)
    fast.set_acc(slow)
    if hasattr(slow, 'include_pn_corrections'):
        #
        slow.set_pnacc(fast)
        fast.set_pnacc(slow)
        slow.pn_kick_ke(dt / 2)
        slow.pn_kick_lmom(dt / 2)
        slow.pn_kick_amom(dt / 2)
        fast.pn_kick_ke(dt / 2)
        fast.pn_kick_lmom(dt / 2)
        fast.pn_kick_amom(dt / 2)

        slow.wx[...] = slow.vx
        slow.wy[...] = slow.vy
        slow.wz[...] = slow.vz
        fast.wx[...] = fast.vx
        fast.wy[...] = fast.vy
        fast.wz[...] = fast.vz
        slow.wx += (slow.ax + slow.pnax) * dt / 2
        slow.wy += (slow.ay + slow.pnay) * dt / 2
        slow.wz += (slow.az + slow.pnaz) * dt / 2
        fast.wx += (fast.ax + fast.pnax) * dt / 2
        fast.wy += (fast.ay + fast.pnay) * dt / 2
        fast.wz += (fast.az + fast.pnaz) * dt / 2

        slow = vw_swap(slow)
        fast = vw_swap(fast)
        slow.set_pnacc(fast)
        fast.set_pnacc(slow)
        slow = vw_swap(slow)
        fast = vw_swap(fast)

        slow.vx += (slow.ax + slow.pnax) * dt
        slow.vy += (slow.ay + slow.pnay) * dt
        slow.vz += (slow.az + slow.pnaz) * dt
        fast.vx += (fast.ax + fast.pnax) * dt
        fast.vy += (fast.ay + fast.pnay) * dt
        fast.vz += (fast.az + fast.pnaz) * dt

        slow.set_pnacc(fast)
        fast.set_pnacc(slow)
        slow.wx += (slow.ax + slow.pnax) * dt / 2
        slow.wy += (slow.ay + slow.pnay) * dt / 2
        slow.wz += (slow.az + slow.pnaz) * dt / 2
        fast.wx += (fast.ax + fast.pnax) * dt / 2
        fast.wy += (fast.ay + fast.pnay) * dt / 2
        fast.wz += (fast.az + fast.pnaz) * dt / 2
        slow.vx[...] = slow.wx
        slow.vy[...] = slow.wy
        slow.vz[...] = slow.wz
        fast.vx[...] = fast.wx
        fast.vy[...] = fast.wy
        fast.vz[...] = fast.wz

        slow.pn_kick_ke(dt / 2)
        slow.pn_kick_lmom(dt / 2)
        slow.pn_kick_amom(dt / 2)
        fast.pn_kick_ke(dt / 2)
        fast.pn_kick_lmom(dt / 2)
        fast.pn_kick_amom(dt / 2)
        #

        #
        # slow.vx += (slow.ax * dt + slow.wx) / 2
        # slow.vy += (slow.ay * dt + slow.wy) / 2
        # slow.vz += (slow.az * dt + slow.wz) / 2
        # fast.vx += (fast.ax * dt + fast.wx) / 2
        # fast.vy += (fast.ay * dt + fast.wy) / 2
        # fast.vz += (fast.az * dt + fast.wz) / 2

        # slow.set_pnacc(fast)
        # fast.set_pnacc(slow)
        # slow.pn_kick_ke(dt)
        # slow.pn_kick_lmom(dt)
        # slow.pn_kick_amom(dt)
        # fast.pn_kick_ke(dt)
        # fast.pn_kick_lmom(dt)
        # fast.pn_kick_amom(dt)
        # slow.wx[...] = 2 * slow.pnax * dt - slow.wx
        # slow.wy[...] = 2 * slow.pnay * dt - slow.wy
        # slow.wz[...] = 2 * slow.pnaz * dt - slow.wz
        # fast.wx[...] = 2 * fast.pnax * dt - fast.wx
        # fast.wy[...] = 2 * fast.pnay * dt - fast.wy
        # fast.wz[...] = 2 * fast.pnaz * dt - fast.wz

        # slow.vx += (slow.ax * dt + slow.wx) / 2
        # slow.vy += (slow.ay * dt + slow.wy) / 2
        # slow.vz += (slow.az * dt + slow.wz) / 2
        # fast.vx += (fast.ax * dt + fast.wx) / 2
        # fast.vy += (fast.ay * dt + fast.wy) / 2
        # fast.vz += (fast.az * dt + fast.wz) / 2
        #
    else:
        slow = kick_n(slow, dt)
        fast = kick_n(fast, dt)
    return slow, fast


class SIAXX(object):
    """

    """
    coefs = (None, None)

    def __init__(self, meth, sia):
        self.meth = meth
        self.sia = sia

    def dkd(self, ips, dt):
        """Standard DKD-type propagator.

        """
        if ips.n <= 2:
            return fewbody_solver(ips, dt)

        ck, cd = self.coefs
        cn = len(cd) - 1

        for i in range(cn):
            ips = drift(ips, cd[i] * dt) if cd[i] else ips
            ips = kick(ips, ck[i] * dt) if ck[i] else ips

        ips = drift(ips, cd[-1] * dt)
        ips = kick(ips, ck[-1] * dt)
        ips = drift(ips, cd[-1] * dt)

        for i in reversed(range(cn)):
            ips = kick(ips, ck[i] * dt) if ck[i] else ips
            ips = drift(ips, cd[i] * dt) if cd[i] else ips

        return ips

    def kdk(self, ips, dt):
        """Standard KDK-type propagator.

        """
        if ips.n <= 2:
            return fewbody_solver(ips, dt)

        cd, ck = self.coefs
        cn = len(cd) - 1

        for i in range(cn):
            ips = kick(ips, ck[i] * dt) if ck[i] else ips
            ips = drift(ips, cd[i] * dt) if cd[i] else ips

        ips = kick(ips, ck[-1] * dt)
        ips = drift(ips, cd[-1] * dt)
        ips = kick(ips, ck[-1] * dt)

        for i in reversed(range(cn)):
            ips = drift(ips, cd[i] * dt) if cd[i] else ips
            ips = kick(ips, ck[i] * dt) if ck[i] else ips

        return ips

    def sf_dkd(self, slow, fast, dt):
        """

        """
        evolve = getattr(self, self.meth)
        recurse = self.sia.recurse

        ck, cd = self.coefs
        cn = len(cd) - 1

        for i in range(cn):
            if cd[i]:
                slow, fast = sf_drift(slow, fast, cd[i] * dt, evolve, recurse)
            if ck[i]:
                slow, fast = sf_kick(slow, fast, ck[i] * dt)

        slow, fast = sf_drift(slow, fast, cd[-1] * dt, evolve, recurse)
        slow, fast = sf_kick(slow, fast, ck[-1] * dt)
        slow, fast = sf_drift(slow, fast, cd[-1] * dt, evolve, recurse)

        for i in reversed(range(cn)):
            if ck[i]:
                slow, fast = sf_kick(slow, fast, ck[i] * dt)
            if cd[i]:
                slow, fast = sf_drift(slow, fast, cd[i] * dt, evolve, recurse)

        return slow, fast

    def sf_kdk(self, slow, fast, dt):
        """

        """
        evolve = getattr(self, self.meth)
        recurse = self.sia.recurse

        cd, ck = self.coefs
        cn = len(cd) - 1

        for i in range(cn):
            if ck[i]:
                slow, fast = sf_kick(slow, fast, ck[i] * dt)
            if cd[i]:
                slow, fast = sf_drift(slow, fast, cd[i] * dt, evolve, recurse)

        slow, fast = sf_kick(slow, fast, ck[-1] * dt)
        slow, fast = sf_drift(slow, fast, cd[-1] * dt, evolve, recurse)
        slow, fast = sf_kick(slow, fast, ck[-1] * dt)

        for i in reversed(range(cn)):
            if cd[i]:
                slow, fast = sf_drift(slow, fast, cd[i] * dt, evolve, recurse)
            if ck[i]:
                slow, fast = sf_kick(slow, fast, ck[i] * dt)

        return slow, fast


#
# SIA21
#
@bind_all(timings)
class SIA21(SIAXX):
    # REF.: Yoshida, Phys. Lett. A 150 (1990)
    coefs = ([1.0],
             [0.5])

    __call__ = SIAXX.sf_dkd


#
# SIA22
#
@bind_all(timings)
class SIA22(SIAXX):
    # REF.: Omelyan, Mryglod & Folk, Comput. Phys. Comm. 151 (2003)
    coefs = ([0.1931833275037836,
              0.6136333449924328],
             [None,
              0.5])

    __call__ = SIAXX.sf_kdk


#
# SIA43
#
@bind_all(timings)
class SIA43(SIAXX):
    # REF.: Yoshida, Phys. Lett. A 150 (1990)
    coefs = ([1.3512071919596575,
              -1.7024143839193150],
             [0.6756035959798288,
              -0.17560359597982877])

    __call__ = SIAXX.sf_dkd


#
# SIA44
#
@bind_all(timings)
class SIA44(SIAXX):
    # REF.: Omelyan, Mryglod & Folk, Comput. Phys. Comm. 151 (2003)
    coefs = ([0.1786178958448091,
              -0.06626458266981843,
              0.7752933736500186],
             [None,
              0.7123418310626056,
              -0.21234183106260562])

    __call__ = SIAXX.sf_kdk


#
# SIA45
#
@bind_all(timings)
class SIA45(SIAXX):
    # REF.: Omelyan, Mryglod & Folk, Comput. Phys. Comm. 151 (2003)
    coefs = ([-0.0844296195070715,
              0.354900057157426,
              0.459059124699291],
             [0.2750081212332419,
              -0.1347950099106792,
              0.35978688867743724])

    __call__ = SIAXX.sf_dkd


#
# SIA46
#
@bind_all(timings)
class SIA46(SIAXX):
    # REF.: Blanes & Moan, J. Comp. Appl. Math. 142 (2002)
    coefs = ([0.0792036964311957,
              0.353172906049774,
              -0.0420650803577195,
              0.21937695575349958],
             [None,
              0.209515106613362,
              -0.143851773179818,
              0.434336666566456])

    __call__ = SIAXX.sf_kdk


#
# SIA67
#
@bind_all(timings)
class SIA67(SIAXX):
    # REF.: Yoshida, Phys. Lett. A 150 (1990)
    coefs = ([0.7845136104775573,
              0.23557321335935813,
              -1.177679984178871,
              1.3151863206839112],
             [0.39225680523877865,
              0.5100434119184577,
              -0.47105338540975644,
              0.06875316825252015])

    __call__ = SIAXX.sf_dkd


#
# SIA69
#
@bind_all(timings)
class SIA69(SIAXX):
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

    __call__ = SIAXX.sf_dkd


@bind_all(timings)
class SIA(Base):
    """

    """
    PROVIDED_METHODS = [
        'sia21s.dkd', 'sia21a.dkd', 'sia21h.dkd',
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

        if method not in self.PROVIDED_METHODS:
            raise ValueError('Invalid integration method: {0}'.format(method))

        if 's.' in method:
            self.update_tstep = False
            self.shared_tstep = True
        elif 'a.' in method:
            self.update_tstep = True
            self.shared_tstep = True
        elif 'h.' in method:
            self.update_tstep = True
            self.shared_tstep = False

        if 'dkd' in method:
            if 'sia21' in method:
                self.bridge = SIA21('dkd', self)
            elif 'sia22' in method:
                self.bridge = SIA22('dkd', self)
            elif 'sia43' in method:
                self.bridge = SIA43('dkd', self)
            elif 'sia44' in method:
                self.bridge = SIA44('dkd', self)
            elif 'sia45' in method:
                self.bridge = SIA45('dkd', self)
            elif 'sia46' in method:
                self.bridge = SIA46('dkd', self)
            elif 'sia67' in method:
                self.bridge = SIA67('dkd', self)
            elif 'sia69' in method:
                self.bridge = SIA69('dkd', self)
        elif 'kdk' in method:
            if 'sia21' in method:
                self.bridge = SIA21('kdk', self)
            elif 'sia22' in method:
                self.bridge = SIA22('kdk', self)
            elif 'sia43' in method:
                self.bridge = SIA43('kdk', self)
            elif 'sia44' in method:
                self.bridge = SIA44('kdk', self)
            elif 'sia45' in method:
                self.bridge = SIA45('kdk', self)
            elif 'sia46' in method:
                self.bridge = SIA46('kdk', self)
            elif 'sia67' in method:
                self.bridge = SIA67('kdk', self)
            elif 'sia69' in method:
                self.bridge = SIA69('kdk', self)

    def initialize(self, t_end):
        """

        """
        ps = self.ps
        LOGGER.info("Initializing '%s' integrator at "
                    "t_curr = %g and t_end = %g.",
                    self.method, ps.t_curr, t_end)

        if hasattr(ps, 'include_pn_corrections'):
            ps.register_attribute('wx', 'real')
            ps.register_attribute('wy', 'real')
            ps.register_attribute('wz', 'real')

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
        ps = self.ps
        LOGGER.info("Finalizing '%s' integrator at "
                    "t_curr = %g and t_end = %g.",
                    self.method, ps.t_curr, t_end)

        if self.viewer:
            self.viewer.show_event(ps)
            self.viewer.enter_main_loop()

    def do_step(self, ps, dt):
        """

        """
        ps = self.recurse(ps, dt)
        type(ps).t_curr += dt
        return ps

    def recurse(self, ps, dt):
        """

        """
        flag = -1
        if self.update_tstep:
            flag = 1
            ps.set_tstep(ps, self.eta)
            if self.shared_tstep:
                dt = self.get_min_block_tstep(ps, dt)

        slow, fast = split(ps, abs(ps.tstep) > flag*abs(dt))

        slow, fast = self.bridge(slow, fast, dt)

        slow = self.dump(dt, slow) if slow.n else slow

        return join(slow, fast)


# -- End of File --
