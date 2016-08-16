# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import numpy as np
from .base import Base
from ..lib import extensions as ext
from ..lib.utils.timing import timings, bind_all


#
# split
#
@timings
def split(ps, mask):
    """Splits the particle's system into slow<->fast components.

    """
    if all(mask):
        return ps, type(ps)()
    if not any(mask):
        return type(ps)(), ps
    return ps.split_by(mask)


#
# join
#
@timings
def join(slow, fast):
    """Joins the slow<->fast components of a particle's system.

    """
    if not fast.n:
        return slow
    if not slow.n:
        return fast
    slow.append(fast)
    return slow


#
# n_drift
#
@timings
def n_drift(ips, dt):
    """Newtonian drift operator.

    """
    ips.time += dt
    ips.rdot[0] += ips.rdot[1] * dt
    return ips


#
# pn_drift
#
@timings
def pn_drift(ips, dt):
    """Post-Newtonian drift operator.

    """
    ips.time += dt
    ips.rdot[0] += ips.rdot[1] * dt
    ips.pn_mr += ips.pn_mv * dt
    return ips


#
# drift
#
@timings
def drift(ips, dt):
    """Drift operator.

    """
    if ips.include_pn_corrections:
        return pn_drift(ips, dt)
    return n_drift(ips, dt)


#
# n_kick
#
@timings
def n_kick(ips, dt):
    """Newtonian kick operator.

    """
    ips.rdot[1] += ips.rdot[2] * dt
    return ips


#
# pn_kick
#
@timings
def pn_kick(ips, dt):
    """Post-Newtonian kick operator.

    """
    #
    ips.set_pnacc(ips)
    ips.pnvel += (ips.rdot[2] + ips.pnacc) * (dt / 2)

    ips.set_pnacc(ips, use_auxvel=True)
    pnforce = ips.mass * ips.pnacc
    ips.pn_mv -= pnforce * dt
    ips.pn_am -= np.cross(ips.rdot[0].T, pnforce.T).T * dt
    ips.pn_ke -= (ips.pnvel * pnforce).sum(0) * dt
    ips.rdot[1] += (ips.rdot[2] + ips.pnacc) * dt

    ips.set_pnacc(ips)
    ips.pnvel += (ips.rdot[2] + ips.pnacc) * (dt / 2)
    #
    return ips


#
# kick
#
@timings
def kick(ips, dt):
    """Kick operator.

    """
    ips.set_acc(ips)
    if ips.include_pn_corrections:
        return pn_kick(ips, dt)
    return n_kick(ips, dt)


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
        ips.time += dt
    return ips


#
# sf_n_kick
#
@timings
def sf_n_kick(slow, fast, dt):
    """Newtonian slow<->fast kick operator.

    """
    for ps in [slow, fast]:
        ps.rdot[1] += ps.rdot[2] * dt
    return slow, fast


#
# sf_pn_kick
#
@timings
def sf_pn_kick(slow, fast, dt):
    """Post-Newtonian slow<->fast kick operator.

    """
    #
    slow.set_pnacc(fast)
    for ps in [slow, fast]:
        ps.pnvel += (ps.rdot[2] + ps.pnacc) * (dt / 2)

    slow.set_pnacc(fast, use_auxvel=True)
    for ps in [slow, fast]:
        pnforce = ps.mass * ps.pnacc
        ps.pn_mv -= pnforce * dt
        ps.pn_am -= np.cross(ps.rdot[0].T, pnforce.T).T * dt
        ps.pn_ke -= (ps.pnvel * pnforce).sum(0) * dt
        ps.rdot[1] += (ps.rdot[2] + ps.pnacc) * dt

    slow.set_pnacc(fast)
    for ps in [slow, fast]:
        ps.pnvel += (ps.rdot[2] + ps.pnacc) * (dt / 2)
    #
    return slow, fast


class SIAXX(object):
    """

    """
    coefs = [(None, None)]

    def __init__(self, manager, meth):
        self.dump = manager.dump
        self.recurse = manager.recurse
        self.shared_tstep = manager.shared_tstep
        self.evolve = getattr(self, meth)
        type(self).__call__ = getattr(self, 'sf_' + meth)

    def dkd(self, ips, dt):
        """Arbitrary order DKD-type operator.

        """
        if ips.n:
            if ips.n == 1:
                ips = drift(ips, dt)
#            elif ips.n == 2:
#                ips = twobody_solver(ips, dt)
            else:
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

            ips.nstep += 1
            if not self.shared_tstep:
                self.dump(ips, dt)

        return ips

    def kdk(self, ips, dt):
        """Arbitrary order KDK-type operator.

        """
        if ips.n:
            if ips.n == 1:
                ips = drift(ips, dt)
#            elif ips.n == 2:
#                ips = twobody_solver(ips, dt)
            else:
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

            ips.nstep += 1
            if not self.shared_tstep:
                self.dump(ips, dt)

        return ips

    def sf_drift(self, slow, fast, dt):
        """Slow<->fast drift operator.

        """
        fast = self.recurse(fast, dt)
        slow = self.evolve(slow, dt)
        return slow, fast

    @staticmethod
    def sf_kick(slow, fast, dt):
        """Slow<->fast kick operator.

        """
        if slow.n and fast.n:
            slow.set_acc(fast)
            if slow.include_pn_corrections:
                return sf_pn_kick(slow, fast, dt)
            return sf_n_kick(slow, fast, dt)
        return slow, fast

    def sf_dkd(self, slow, fast, dt):
        """Arbitrary order slow<->fast DKD-type operator.

        """
        if not fast.n:
            slow = self.evolve(slow, dt)
            return slow, fast

        coefs = self.coefs
        sf_kick = self.sf_kick
        sf_drift = self.sf_drift
        ck0, cd0 = coefs[-1]

        for ck, cd in coefs[:-1]:
            slow, fast = sf_drift(slow, fast, cd * dt) if cd else (slow, fast)
            slow, fast = sf_kick(slow, fast, ck * dt) if ck else (slow, fast)

        slow, fast = sf_drift(slow, fast, cd0 * dt)
        slow, fast = sf_kick(slow, fast, ck0 * dt)
        slow, fast = sf_drift(slow, fast, cd0 * dt)

        for ck, cd in reversed(coefs[:-1]):
            slow, fast = sf_kick(slow, fast, ck * dt) if ck else (slow, fast)
            slow, fast = sf_drift(slow, fast, cd * dt) if cd else (slow, fast)

        return slow, fast

    def sf_kdk(self, slow, fast, dt):
        """Arbitrary order slow<->fast KDK-type operator.

        """
        if not fast.n:
            slow = self.evolve(slow, dt)
            return slow, fast

        coefs = self.coefs
        sf_kick = self.sf_kick
        sf_drift = self.sf_drift
        cd0, ck0 = coefs[-1]

        for cd, ck in coefs[:-1]:
            slow, fast = sf_kick(slow, fast, ck * dt) if ck else (slow, fast)
            slow, fast = sf_drift(slow, fast, cd * dt) if cd else (slow, fast)

        slow, fast = sf_kick(slow, fast, ck0 * dt)
        slow, fast = sf_drift(slow, fast, cd0 * dt)
        slow, fast = sf_kick(slow, fast, ck0 * dt)

        for cd, ck in reversed(coefs[:-1]):
            slow, fast = sf_drift(slow, fast, cd * dt) if cd else (slow, fast)
            slow, fast = sf_kick(slow, fast, ck * dt) if ck else (slow, fast)

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

    def __init__(self, ps, eta, dt_max, t_begin, method, **kwargs):
        """

        """
        super(SIA, self).__init__(ps, eta, dt_max,
                                  t_begin, method, **kwargs)

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

        ps.tstep[...] = dt_max
        if self.update_tstep:
            ps.set_tstep(ps, self.eta)
            if self.shared_tstep:
                ps.tstep[...] = ps.tstep.min()

    def dump(self, ps, dt):
        """

        """
        tdiff = abs(ps.t_next - ps.time[0])
        ratio = tdiff // abs(dt)
        s0 = tdiff > 0
        if self.dumpper:
            s1 = ratio % self.dump_freq == 0
            if (s0 and s1):
                self.dumpper.append_data(ps)
        if self.viewer:
            s1 = ratio % self.viewer.gl_freq == 0
            if (s0 and s1):
                self.viewer.show_event(ps)

    def do_step(self, ps, dt):
        """

        """
        if ps.include_pn_corrections:
            if not hasattr(ps, 'pnvel'):
                ps.register_attribute('pnvel', '{nd}, {nb}', 'real_t')
                ps.pnvel[...] = ps.rdot[1]

        return self.recurse(ps, dt)

    def recurse(self, ps, dt):
        """

        """
        if ps.n:
            slow, fast = split(ps, ps.tstep > abs(dt))
            slow, fast = self.bridge(slow, fast, dt)
            ps = join(slow, fast)
            if self.update_tstep:
                ps.set_tstep(ps, self.eta)
                if self.shared_tstep:
                    ps.tstep[...] = ps.tstep.min()
        return ps


# -- End of File --
