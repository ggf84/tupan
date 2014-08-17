# -*- coding: utf-8 -*-
#

"""
TODO.
"""


import logging
from .base import Base, power_of_two
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
    if all(condition):
        return ps, type(ps)()

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
    ips.pos += ips.vel * dt
    return ips


#
# kick_n
#
@timings
def kick_n(ips, dt):
    """Kick operator for Newtonian quantities.

    """
    ips.vel += ips.acc * dt
    return ips


#
# drift_pn
#
@timings
def drift_pn(ips, dt):
    """Drift operator for post-Newtonian quantities.

    """
    ips.pos += ips.vel * dt
    ips.pn_drift_com_r(dt)
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

    ips.w[...] = ips.vel
    ips.w += (ips.acc + ips.pnacc) * dt / 2

    ips.vel, ips.w = ips.w, ips.vel
    ips.set_pnacc(ips)
    ips.vel, ips.w = ips.w, ips.vel

    ips.vel += (ips.acc + ips.pnacc) * dt

    ips.set_pnacc(ips)
    ips.w += (ips.acc + ips.pnacc) * dt / 2
    ips.vel[...] = ips.w

    ips.pn_kick_ke(dt / 2)
    ips.pn_kick_lmom(dt / 2)
    ips.pn_kick_amom(dt / 2)
    #

    #
    # ips.vel += (ips.acc * dt + ips.w) / 2

    # ips.set_pnacc(ips)
    # ips.pn_kick_ke(dt)
    # ips.pn_kick_lmom(dt)
    # ips.pn_kick_amom(dt)
    # ips.w[...] = 2 * ips.pnacc * dt - ips.w

    # ips.vel += (ips.acc * dt + ips.w) / 2
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

        slow.w[...] = slow.vel
        fast.w[...] = fast.vel
        slow.w += (slow.acc + slow.pnacc) * dt / 2
        fast.w += (fast.acc + fast.pnacc) * dt / 2

        slow.vel, slow.w = slow.w, slow.vel
        fast.vel, fast.w = fast.w, fast.vel
        slow.set_pnacc(fast)
        fast.set_pnacc(slow)
        slow.vel, slow.w = slow.w, slow.vel
        fast.vel, fast.w = fast.w, fast.vel

        slow.vel += (slow.acc + slow.pnacc) * dt
        fast.vel += (fast.acc + fast.pnacc) * dt

        slow.set_pnacc(fast)
        fast.set_pnacc(slow)
        slow.w += (slow.acc + slow.pnacc) * dt / 2
        fast.w += (fast.acc + fast.pnacc) * dt / 2
        slow.vel[...] = slow.w
        fast.vel[...] = fast.w

        slow.pn_kick_ke(dt / 2)
        slow.pn_kick_lmom(dt / 2)
        slow.pn_kick_amom(dt / 2)
        fast.pn_kick_ke(dt / 2)
        fast.pn_kick_lmom(dt / 2)
        fast.pn_kick_amom(dt / 2)
        #

        #
        # slow.vel += (slow.acc * dt + slow.w) / 2
        # fast.vel += (fast.acc * dt + fast.w) / 2

        # slow.set_pnacc(fast)
        # fast.set_pnacc(slow)
        # slow.pn_kick_ke(dt)
        # slow.pn_kick_lmom(dt)
        # slow.pn_kick_amom(dt)
        # fast.pn_kick_ke(dt)
        # fast.pn_kick_lmom(dt)
        # fast.pn_kick_amom(dt)
        # slow.w[...] = 2 * slow.pnacc * dt - slow.w
        # fast.w[...] = 2 * fast.pnacc * dt - fast.w

        # slow.vel += (slow.acc * dt + slow.w) / 2
        # fast.vel += (fast.acc * dt + fast.w) / 2
        #
    else:
        slow = kick_n(slow, dt)
        fast = kick_n(fast, dt)
    return slow, fast


class SIAXX(object):
    """

    """
    coefs = (None, None)

    def __init__(self, manager, meth):
        self.manager = manager
        self.meth = meth

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
        recurse = self.manager.recurse

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

        if fast.n == 0:
            type(slow).t_curr += dt

        if slow.n:
            slow.tstep[...] = dt
            slow.time += dt
            slow.nstep += 1
            self.manager.dump(dt, slow)

        return slow, fast

    def sf_kdk(self, slow, fast, dt):
        """

        """
        evolve = getattr(self, self.meth)
        recurse = self.manager.recurse

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

        if fast.n == 0:
            type(slow).t_curr += dt

        if slow.n:
            slow.tstep[...] = dt
            slow.time += dt
            slow.nstep += 1
            self.manager.dump(dt, slow)

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
        'sia21c.dkd', 'sia21a.dkd', 'sia21h.dkd',
        'sia21c.kdk', 'sia21a.kdk', 'sia21h.kdk',
        'sia22c.dkd', 'sia22a.dkd', 'sia22h.dkd',
        'sia22c.kdk', 'sia22a.kdk', 'sia22h.kdk',
        'sia43c.dkd', 'sia43a.dkd', 'sia43h.dkd',
        'sia43c.kdk', 'sia43a.kdk', 'sia43h.kdk',
        'sia44c.dkd', 'sia44a.dkd', 'sia44h.dkd',
        'sia44c.kdk', 'sia44a.kdk', 'sia44h.kdk',
        'sia45c.dkd', 'sia45a.dkd', 'sia45h.dkd',
        'sia45c.kdk', 'sia45a.kdk', 'sia45h.kdk',
        'sia46c.dkd', 'sia46a.dkd', 'sia46h.dkd',
        'sia46c.kdk', 'sia46a.kdk', 'sia46h.kdk',
        'sia67c.dkd', 'sia67a.dkd', 'sia67h.dkd',
        'sia67c.kdk', 'sia67a.kdk', 'sia67h.kdk',
        'sia69c.dkd', 'sia69a.dkd', 'sia69h.dkd',
        'sia69c.kdk', 'sia69a.kdk', 'sia69h.kdk',
    ]

    def __init__(self, eta, time, ps, method, **kwargs):
        """

        """
        super(SIA, self).__init__(eta, time, ps, **kwargs)
        self.method = method

        if method not in self.PROVIDED_METHODS:
            raise ValueError('Invalid integration method: {0}'.format(method))

        if 'c.' in method:
            self.update_tstep = False
        elif 'a.' in method:
            self.update_tstep = True
            self.shared_tstep = True
        elif 'h.' in method:
            self.update_tstep = True
            self.shared_tstep = False

        if 'dkd' in method:
            if 'sia21' in method:
                self.bridge = SIA21(self, 'dkd')
            elif 'sia22' in method:
                self.bridge = SIA22(self, 'dkd')
            elif 'sia43' in method:
                self.bridge = SIA43(self, 'dkd')
            elif 'sia44' in method:
                self.bridge = SIA44(self, 'dkd')
            elif 'sia45' in method:
                self.bridge = SIA45(self, 'dkd')
            elif 'sia46' in method:
                self.bridge = SIA46(self, 'dkd')
            elif 'sia67' in method:
                self.bridge = SIA67(self, 'dkd')
            elif 'sia69' in method:
                self.bridge = SIA69(self, 'dkd')
        elif 'kdk' in method:
            if 'sia21' in method:
                self.bridge = SIA21(self, 'kdk')
            elif 'sia22' in method:
                self.bridge = SIA22(self, 'kdk')
            elif 'sia43' in method:
                self.bridge = SIA43(self, 'kdk')
            elif 'sia44' in method:
                self.bridge = SIA44(self, 'kdk')
            elif 'sia45' in method:
                self.bridge = SIA45(self, 'kdk')
            elif 'sia46' in method:
                self.bridge = SIA46(self, 'kdk')
            elif 'sia67' in method:
                self.bridge = SIA67(self, 'kdk')
            elif 'sia69' in method:
                self.bridge = SIA69(self, 'kdk')

    def initialize(self, t_end):
        """

        """
        ps = self.ps
        LOGGER.info("Initializing '%s' integrator at "
                    "t_curr = %g and t_end = %g.",
                    self.method, ps.t_curr, t_end)

        if hasattr(ps, 'include_pn_corrections'):
            ps.register_attribute('w', (3,), 'real')

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
        return self.recurse(ps, dt)

    def recurse(self, ps, dt):
        """

        """
        if self.update_tstep:
            ps.set_tstep(ps, self.eta)
            if self.shared_tstep:
                dt = power_of_two(ps, dt)
            condition = abs(ps.tstep) > abs(dt)
        else:
            condition = abs(ps.tstep) > -1

        slow, fast = split(ps, condition)
        slow, fast = self.bridge(slow, fast, dt)
        return join(slow, fast)


# -- End of File --
