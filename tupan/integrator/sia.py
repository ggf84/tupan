# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import logging
import numpy as np
from .base import Base, power_of_two
from ..lib import extensions as ext
from ..lib.utils.timing import timings, bind_all


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

    if all(condition):
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
    ips.pn_mr += ips.pn_mv * dt
    return ips


#
# kick_pn
#
@timings
def kick_pn(ips, dt):
    """Kick operator for post-Newtonian quantities.

    """
    #
    ips.wel[...] = ips.vel

    ips.set_pnacc(ips, use_auxvel=True)
    pnforce = ips.mass * ips.pnacc
    ips.vel += (ips.acc + ips.pnacc) * (dt / 2)
    ips.pn_ke -= (ips.wel * pnforce).sum(0) * (dt / 2)
    ips.pn_mv -= pnforce * (dt / 2)
    ips.pn_am -= np.cross(ips.pos.T, pnforce.T).T * (dt / 2)

    ips.set_pnacc(ips)
    ips.wel += (ips.acc + ips.pnacc) * dt

    ips.set_pnacc(ips, use_auxvel=True)
    pnforce = ips.mass * ips.pnacc
    ips.vel += (ips.acc + ips.pnacc) * (dt / 2)
    ips.pn_ke -= (ips.wel * pnforce).sum(0) * (dt / 2)
    ips.pn_mv -= pnforce * (dt / 2)
    ips.pn_am -= np.cross(ips.pos.T, pnforce.T).T * (dt / 2)
    #

    #
    # ips.vel += (ips.acc * dt + ips.wel) / 2
    #
    # ips.set_pnacc(ips)
    # pnforce = ips.mass * ips.pnacc
    # ips.pn_ke -= (ips.vel * pnforce).sum(0) * dt
    # ips.pn_mv -= pnforce * dt
    # ips.pn_am -= np.cross(ips.pos.T, pnforce.T).T * dt
    # ips.wel[...] = 2 * ips.pnacc * dt - ips.wel
    #
    # ips.vel += (ips.acc * dt + ips.wel) / 2
    #

    return ips


#
# drift
#
@timings
def drift(ips, dt):
    """Drift operator.

    """
    if ips.include_pn_corrections:
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
    if ips.include_pn_corrections:
        return kick_pn(ips, dt)
    return kick_n(ips, dt)


#
# twobody_solver
#
@timings
def twobody_solver(ips, dt, kernel=ext.Kepler()):
    """

    """
    if ips.include_pn_corrections:
        raise NotImplementedError('The current version of the '
                                  'Kepler-solver does not include '
                                  'post-Newtonian corrections.')
    else:
        ps0, ps1 = ips[0], ips[1]
        kernel(next(iter(ps0.members.values())),
               next(iter(ps1.members.values())),
               dt=dt)
        ips = join(ps0, ps1)
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
    fast = recurse(fast, 0.5 * dt) if fast.n else fast
    slow = evolve(slow, dt) if slow.n else slow
    fast = recurse(fast, 0.5 * dt) if fast.n else fast
    return slow, fast


#
# sf_kick_n
#
@timings
def sf_kick_n(slow, fast, dt):
    """Slow<->Fast Kick operator for Newtonian quantities.

    """
    slow = kick_n(slow, dt)
    fast = kick_n(fast, dt)
    return slow, fast


#
# sf_kick_pn
#
@timings
def sf_kick_pn(slow, fast, dt):
    """Slow<->Fast Kick operator for post-Newtonian quantities.

    """
    #
    for ps in [slow, fast]:
        ps.wel[...] = ps.vel

    slow.set_pnacc(fast, use_auxvel=True)
    fast.set_pnacc(slow, use_auxvel=True)
    for ps in [slow, fast]:
        pnforce = ps.mass * ps.pnacc
        ps.vel += (ps.acc + ps.pnacc) * (dt / 2)
        ps.pn_ke -= (ps.wel * pnforce).sum(0) * (dt / 2)
        ps.pn_mv -= pnforce * (dt / 2)
        ps.pn_am -= np.cross(ps.pos.T, pnforce.T).T * (dt / 2)

    slow.set_pnacc(fast)
    fast.set_pnacc(slow)
    for ps in [slow, fast]:
        ps.wel += (ps.acc + ps.pnacc) * dt

    slow.set_pnacc(fast, use_auxvel=True)
    fast.set_pnacc(slow, use_auxvel=True)
    for ps in [slow, fast]:
        pnforce = ps.mass * ps.pnacc
        ps.vel += (ps.acc + ps.pnacc) * (dt / 2)
        ps.pn_ke -= (ps.wel * pnforce).sum(0) * (dt / 2)
        ps.pn_mv -= pnforce * (dt / 2)
        ps.pn_am -= np.cross(ps.pos.T, pnforce.T).T * (dt / 2)
    #

    #
    # for ps in [slow, fast]:
    #    ps.vel += (ps.acc * dt + ps.wel) / 2
    #
    # slow.set_pnacc(fast)
    # fast.set_pnacc(slow)
    # for ps in [slow, fast]:
    #     pnforce = ps.mass * ps.pnacc
    #     ps.pn_ke -= (ps.vel * pnforce).sum(0) * dt
    #     ps.pn_mv -= pnforce * dt
    #     ps.pn_am -= np.cross(ps.pos.T, pnforce.T).T * dt
    #     ps.wel[...] = 2 * ps.pnacc * dt - ps.wel
    #
    # for ps in [slow, fast]:
    #     ps.vel += (ps.acc * dt + ps.wel) / 2
    #

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
    if slow.include_pn_corrections:
        return sf_kick_pn(slow, fast, dt)
    return sf_kick_n(slow, fast, dt)


class SIAXX(object):
    """

    """
    coefs = [(None, None)]

    def __init__(self, manager, meth):
        self.manager = manager
        type(self).__call__ = getattr(self, 'sf_' + meth)

    def dkd(self, ips, dt):
        """Standard DKD-type propagator.

        """
        if ips.n <= 2:
            return fewbody_solver(ips, dt)

        coefs = self.coefs
        ck0, cd0 = coefs[-1]

        for ck, cd in coefs[:-1]:
            ips = drift(ips, cd * dt) if cd else ips
            ips = kick(ips, ck * dt) if ck else ips

        ips = drift(ips, cd0 * dt)
        ips = kick(ips, ck0 * dt)
        ips = drift(ips, cd0 * dt)

        for ck, cd in reversed(coefs[:-1]):
            ips = kick(ips, ck * dt) if ck else ips
            ips = drift(ips, cd * dt) if cd else ips

        return ips

    def kdk(self, ips, dt):
        """Standard KDK-type propagator.

        """
        if ips.n <= 2:
            return fewbody_solver(ips, dt)

        coefs = self.coefs
        cd0, ck0 = coefs[-1]

        for cd, ck in coefs[:-1]:
            ips = kick(ips, ck * dt) if ck else ips
            ips = drift(ips, cd * dt) if cd else ips

        ips = kick(ips, ck0 * dt)
        ips = drift(ips, cd0 * dt)
        ips = kick(ips, ck0 * dt)

        for cd, ck in reversed(coefs[:-1]):
            ips = drift(ips, cd * dt) if cd else ips
            ips = kick(ips, ck * dt) if ck else ips

        return ips

    def sf_dkd(self, slow, fast, dt):
        """

        """
        evolve = self.dkd
        recurse = self.manager.recurse

        if fast.n:
            coefs = self.coefs
            ck0, cd0 = coefs[-1]

            for ck, cd in coefs[:-1]:
                slow, fast = (sf_drift(slow, fast, cd * dt, evolve, recurse)
                              if cd else (slow, fast))
                slow, fast = (sf_kick(slow, fast, ck * dt)
                              if ck else (slow, fast))

            slow, fast = sf_drift(slow, fast, cd0 * dt, evolve, recurse)
            slow, fast = sf_kick(slow, fast, ck0 * dt)
            slow, fast = sf_drift(slow, fast, cd0 * dt, evolve, recurse)

            for ck, cd in reversed(coefs[:-1]):
                slow, fast = (sf_kick(slow, fast, ck * dt)
                              if ck else (slow, fast))
                slow, fast = (sf_drift(slow, fast, cd * dt, evolve, recurse)
                              if cd else (slow, fast))
        else:
            slow = evolve(slow, dt) if slow.n else slow
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
        evolve = self.kdk
        recurse = self.manager.recurse

        if fast.n:
            coefs = self.coefs
            cd0, ck0 = coefs[-1]

            for cd, ck in coefs[:-1]:
                slow, fast = (sf_kick(slow, fast, ck * dt)
                              if ck else (slow, fast))
                slow, fast = (sf_drift(slow, fast, cd * dt, evolve, recurse)
                              if cd else (slow, fast))

            slow, fast = sf_kick(slow, fast, ck0 * dt)
            slow, fast = sf_drift(slow, fast, cd0 * dt, evolve, recurse)
            slow, fast = sf_kick(slow, fast, ck0 * dt)

            for cd, ck in reversed(coefs[:-1]):
                slow, fast = (sf_drift(slow, fast, cd * dt, evolve, recurse)
                              if cd else (slow, fast))
                slow, fast = (sf_kick(slow, fast, ck * dt)
                              if ck else (slow, fast))
        else:
            slow = evolve(slow, dt) if slow.n else slow
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
    coefs = [(1.0, 0.5)]


#
# SIA22
#
@bind_all(timings)
class SIA22(SIAXX):
    # REF.: Omelyan, Mryglod & Folk, Comput. Phys. Comm. 151 (2003)
    coefs = [(0.1931833275037836, None),
             (0.6136333449924328, 0.5)]


#
# SIA43
#
@bind_all(timings)
class SIA43(SIAXX):
    # REF.: Yoshida, Phys. Lett. A 150 (1990)
    coefs = [(1.3512071919596575, 0.6756035959798288),
             (-1.702414383919315, -0.17560359597982877)]


#
# SIA44
#
@bind_all(timings)
class SIA44(SIAXX):
    # REF.: Omelyan, Mryglod & Folk, Comput. Phys. Comm. 151 (2003)
    coefs = [(0.1786178958448091, None),
             (-0.06626458266981843, 0.7123418310626056),
             (0.7752933736500186, -0.21234183106260562)]


#
# SIA45
#
@bind_all(timings)
class SIA45(SIAXX):
    # REF.: Omelyan, Mryglod & Folk, Comput. Phys. Comm. 151 (2003)
    coefs = [(-0.0844296195070715, 0.2750081212332419),
             (0.354900057157426, -0.1347950099106792),
             (0.459059124699291, 0.35978688867743724)]


#
# SIA46
#
@bind_all(timings)
class SIA46(SIAXX):
    # REF.: Blanes & Moan, J. Comp. Appl. Math. 142 (2002)
    coefs = [(0.0792036964311957, None),
             (0.353172906049774, 0.209515106613362),
             (-0.0420650803577195, -0.143851773179818),
             (0.21937695575349958, 0.434336666566456)]


#
# SIA67
#
@bind_all(timings)
class SIA67(SIAXX):
    # REF.: Yoshida, Phys. Lett. A 150 (1990)
    coefs = [(0.7845136104775573, 0.39225680523877865),
             (0.23557321335935813, 0.5100434119184577),
             (-1.177679984178871, -0.47105338540975644),
             (1.3151863206839112, 0.06875316825252015)]


#
# SIA69
#
@bind_all(timings)
class SIA69(SIAXX):
    # REF.: Kahan & Li, Math. Comput. 66 (1997)
    coefs = [(0.39103020330868477, 0.19551510165434238),
             (0.334037289611136, 0.3625337464599104),
             (-0.7062272811875614, -0.1860949957882127),
             (0.08187754964805945, -0.31217486576975095),
             (0.7985644772393624, 0.44022101344371095)]


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
            self.shared_tstep = True
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

        if ps.include_pn_corrections:
            ps.register_attribute('wel', '3, {n}', 'real_t',
                                  doc='auxiliary-velocity for PN integration.')

        if self.reporter:
            self.reporter.diagnostic_report(ps)
        if self.dumpper:
            self.dumpper.init_worldline(ps)
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
