# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import numpy as np
from .base import Base
from ..lib import extensions as ext


#
# n_drift
#
def n_drift(ips, dt):
    """Newtonian drift operator.

    """
    ips.time += dt
    ips.rdot[0] += ips.rdot[1] * dt
    return ips


#
# pn_drift
#
def pn_drift(ips, dt):
    """Post-Newtonian drift operator.

    """
    ips.time += dt
    ips.rdot[0] += ips.rdot[1] * dt
    ips.pn_mr += ips.pn_mv * dt
    return ips


#
# n_kick
#
def n_kick(ips, dt):
    """Newtonian kick operator.

    """
    ips.rdot[1] += ips.rdot[2] * dt
    return ips


#
# pn_kick
#
def pn_kick(ips, dt, pn=None):
    """Post-Newtonian kick operator.

    """
    #
    ips.set_pnacc(ips, pn=pn)
    ips.pnvel += (ips.rdot[2] + ips.pnacc) * (dt / 2)

    ips.set_pnacc(ips, pn=pn, use_auxvel=True)
    pnforce = ips.mass * ips.pnacc
    ips.pn_mv -= pnforce * dt
    ips.pn_am -= np.cross(ips.rdot[0].T, pnforce.T).T * dt
    ips.pn_ke -= (ips.pnvel * pnforce).sum(0) * dt
    ips.rdot[1] += (ips.rdot[2] + ips.pnacc) * dt

    ips.set_pnacc(ips, pn=pn)
    ips.pnvel += (ips.rdot[2] + ips.pnacc) * (dt / 2)
    #
    return ips


#
# sf_n_kick
#
def sf_n_kick(slow, fast, dt):
    """Newtonian slow<->fast kick operator.

    """
    for ps in [slow, fast]:
        ps.rdot[1] += ps.rdot[2] * dt
    return slow, fast


#
# sf_pn_kick
#
def sf_pn_kick(slow, fast, dt, pn=None):
    """Post-Newtonian slow<->fast kick operator.

    """
    #
    slow.set_pnacc(fast, pn=pn)
    for ps in [slow, fast]:
        ps.pnvel += (ps.rdot[2] + ps.pnacc) * (dt / 2)

    slow.set_pnacc(fast, pn=pn, use_auxvel=True)
    for ps in [slow, fast]:
        pnforce = ps.mass * ps.pnacc
        ps.pn_mv -= pnforce * dt
        ps.pn_am -= np.cross(ps.rdot[0].T, pnforce.T).T * dt
        ps.pn_ke -= (ps.pnvel * pnforce).sum(0) * dt
        ps.rdot[1] += (ps.rdot[2] + ps.pnacc) * dt

    slow.set_pnacc(fast, pn=pn)
    for ps in [slow, fast]:
        ps.pnvel += (ps.rdot[2] + ps.pnacc) * (dt / 2)
    #
    return slow, fast


#
# twobody_solver
#
def twobody_solver(ips, dt, pn=None, kernel=ext.Kepler()):
    """

    """
    if pn:
        raise NotImplementedError('The current version of the '
                                  'Kepler-solver does not include '
                                  'post-Newtonian corrections.')
    else:
        ps0, ps1 = ips[0], ips[1]
        kernel(next(iter(ps0.members.values())),
               next(iter(ps1.members.values())),
               dt=dt)
        ips = ps0 + ps1
        ips.time += dt
    return ips


class SIAXX(object):
    """

    """
    coefs = [(None, None)]

    def __init__(self, manager, meth):
        self.pn = manager.pn
        self.eta = manager.eta
        self.dump = manager.dump
        self.update_tstep = manager.update_tstep
        self.shared_tstep = manager.shared_tstep
        self.evolve = getattr(self, meth)
        self.bridge = getattr(self, 'sf_' + meth)

    def drift(self, ips, dt):
        """Drift operator.

        """
        if self.pn:
            return pn_drift(ips, dt)
        return n_drift(ips, dt)

    def kick(self, ips, dt):
        """Kick operator.

        """
        ips.set_acc(ips)
        if self.pn:
            return pn_kick(ips, dt, pn=self.pn)
        return n_kick(ips, dt)

    def dkd(self, ips, dt):
        """Arbitrary order DKD-type operator.

        """
        kick = self.kick
        drift = self.drift

        if ips.n == 1:
            return drift(ips, dt)

#        if ips.n == 2:
#            return twobody_solver(ips, dt, pn=self.pn)

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
        """Arbitrary order KDK-type operator.

        """
        kick = self.kick
        drift = self.drift

        if ips.n == 1:
            return drift(ips, dt)

#        if ips.n == 2:
#            return twobody_solver(ips, dt, pn=self.pn)

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

    def recurse(self, ps, dt):
        """

        """
        slow, fast = ps.split_by(lambda obj: obj.tstep > abs(dt))
        ps = self.bridge(slow, fast, dt)
        if self.update_tstep:
            ps.set_tstep(ps, self.eta)
            if self.shared_tstep:
                ps.tstep[...] = ps.tstep.min()
        return ps

    def sf_drift(self, slow, fast, dt):
        """Slow<->fast drift operator.

        """
        fast = self.recurse(fast, dt)
        slow = self.evolve(slow, dt)
        return slow, fast

    def sf_kick(self, slow, fast, dt):
        """Slow<->fast kick operator.

        """
        slow.set_acc(fast)
        if self.pn:
            return sf_pn_kick(slow, fast, dt, pn=self.pn)
        return sf_n_kick(slow, fast, dt)

    def sf_dkd(self, slow, fast, dt):
        """Arbitrary order slow<->fast DKD-type operator.

        """
        if not fast.n:
            slow = self.evolve(slow, dt)
            slow.nstep += 1
            self.dump(slow, dt)
            return slow

        if not slow.n:
            fast = self.recurse(fast, dt/2)
            fast = self.recurse(fast, dt/2)
            return fast

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

        slow.nstep += 1
        self.dump(slow, dt)
        return slow + fast

    def sf_kdk(self, slow, fast, dt):
        """Arbitrary order slow<->fast KDK-type operator.

        """
        if not fast.n:
            slow = self.evolve(slow, dt)
            slow.nstep += 1
            self.dump(slow, dt)
            return slow

        if not slow.n:
            fast = self.recurse(fast, dt/2)
            fast = self.recurse(fast, dt/2)
            return fast

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

        slow.nstep += 1
        self.dump(slow, dt)
        return slow + fast


#
# SIA21
#
class SIA21(SIAXX):
    # REF.: Yoshida, Phys. Lett. A 150 (1990)
    coefs = [(1.0, 0.5)]


#
# SIA22
#
class SIA22(SIAXX):
    # REF.: Omelyan, Mryglod & Folk, Comput. Phys. Comm. 151 (2003)
    coefs = [(0.1931833275037836, None),
             (0.6136333449924328, 0.5)]


#
# SIA43
#
class SIA43(SIAXX):
    # REF.: Yoshida, Phys. Lett. A 150 (1990)
    coefs = [(1.3512071919596575, 0.6756035959798288),
             (-1.702414383919315, -0.17560359597982877)]


#
# SIA44
#
class SIA44(SIAXX):
    # REF.: Omelyan, Mryglod & Folk, Comput. Phys. Comm. 151 (2003)
    coefs = [(0.1786178958448091, None),
             (-0.06626458266981843, 0.7123418310626056),
             (0.7752933736500186, -0.21234183106260562)]


#
# SIA45
#
class SIA45(SIAXX):
    # REF.: Omelyan, Mryglod & Folk, Comput. Phys. Comm. 151 (2003)
    coefs = [(-0.0844296195070715, 0.2750081212332419),
             (0.354900057157426, -0.1347950099106792),
             (0.459059124699291, 0.35978688867743724)]


#
# SIA46
#
class SIA46(SIAXX):
    # REF.: Blanes & Moan, J. Comp. Appl. Math. 142 (2002)
    coefs = [(0.0792036964311957, None),
             (0.353172906049774, 0.209515106613362),
             (-0.0420650803577195, -0.143851773179818),
             (0.21937695575349958, 0.434336666566456)]


#
# SIA67
#
class SIA67(SIAXX):
    # REF.: Yoshida, Phys. Lett. A 150 (1990)
    coefs = [(0.7845136104775573, 0.39225680523877865),
             (0.23557321335935813, 0.5100434119184577),
             (-1.177679984178871, -0.47105338540975644),
             (1.3151863206839112, 0.06875316825252015)]


#
# SIA69
#
class SIA69(SIAXX):
    # REF.: Kahan & Li, Math. Comput. 66 (1997)
    coefs = [(0.39103020330868477, 0.19551510165434238),
             (0.334037289611136, 0.3625337464599104),
             (-0.7062272811875614, -0.1860949957882127),
             (0.08187754964805945, -0.31217486576975095),
             (0.7985644772393624, 0.44022101344371095)]


class SIA(Base):
    """

    """
    PROVIDED_METHODS = [
        'c.sia21.dkd', 'a.sia21.dkd', 'h.sia21.dkd',
        'c.sia21.kdk', 'a.sia21.kdk', 'h.sia21.kdk',
        'c.sia22.dkd', 'a.sia22.dkd', 'h.sia22.dkd',
        'c.sia22.kdk', 'a.sia22.kdk', 'h.sia22.kdk',
        'c.sia43.dkd', 'a.sia43.dkd', 'h.sia43.dkd',
        'c.sia43.kdk', 'a.sia43.kdk', 'h.sia43.kdk',
        'c.sia44.dkd', 'a.sia44.dkd', 'h.sia44.dkd',
        'c.sia44.kdk', 'a.sia44.kdk', 'h.sia44.kdk',
        'c.sia45.dkd', 'a.sia45.dkd', 'h.sia45.dkd',
        'c.sia45.kdk', 'a.sia45.kdk', 'h.sia45.kdk',
        'c.sia46.dkd', 'a.sia46.dkd', 'h.sia46.dkd',
        'c.sia46.kdk', 'a.sia46.kdk', 'h.sia46.kdk',
        'c.sia67.dkd', 'a.sia67.dkd', 'h.sia67.dkd',
        'c.sia67.kdk', 'a.sia67.kdk', 'h.sia67.kdk',
        'c.sia69.dkd', 'a.sia69.dkd', 'h.sia69.dkd',
        'c.sia69.kdk', 'a.sia69.kdk', 'h.sia69.kdk',
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
                self.sia = SIA21(self, 'dkd')
            elif 'sia22' in method:
                self.sia = SIA22(self, 'dkd')
            elif 'sia43' in method:
                self.sia = SIA43(self, 'dkd')
            elif 'sia44' in method:
                self.sia = SIA44(self, 'dkd')
            elif 'sia45' in method:
                self.sia = SIA45(self, 'dkd')
            elif 'sia46' in method:
                self.sia = SIA46(self, 'dkd')
            elif 'sia67' in method:
                self.sia = SIA67(self, 'dkd')
            elif 'sia69' in method:
                self.sia = SIA69(self, 'dkd')
        elif 'kdk' in method:
            if 'sia21' in method:
                self.sia = SIA21(self, 'kdk')
            elif 'sia22' in method:
                self.sia = SIA22(self, 'kdk')
            elif 'sia43' in method:
                self.sia = SIA43(self, 'kdk')
            elif 'sia44' in method:
                self.sia = SIA44(self, 'kdk')
            elif 'sia45' in method:
                self.sia = SIA45(self, 'kdk')
            elif 'sia46' in method:
                self.sia = SIA46(self, 'kdk')
            elif 'sia67' in method:
                self.sia = SIA67(self, 'kdk')
            elif 'sia69' in method:
                self.sia = SIA69(self, 'kdk')

        if self.pn:
            if not hasattr(ps, 'pn_ke'):
                ps.register_attribute('pn_ke', '{nb}', 'real_t')
            if not hasattr(ps, 'pn_mr'):
                ps.register_attribute('pn_mr', '{nd}, {nb}', 'real_t')
            if not hasattr(ps, 'pn_mv'):
                ps.register_attribute('pn_mv', '{nd}, {nb}', 'real_t')
            if not hasattr(ps, 'pn_am'):
                ps.register_attribute('pn_am', '{nd}, {nb}', 'real_t')
            if not hasattr(ps, 'pnacc'):
                ps.register_attribute('pnacc', '{nd}, {nb}', 'real_t')
            if not hasattr(ps, 'pnvel'):
                ps.register_attribute('pnvel', '{nd}, {nb}', 'real_t')
                ps.pnvel[...] = ps.rdot[1]

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
        if self.dumpper:
            s0 = tdiff > 0
            s1 = ratio % self.dump_freq == 0
            if (s0 and s1):
                self.dumpper.append_data(ps)
        if self.viewer:
            s1 = ratio % self.viewer.gl_freq == 0
            if s1:
                self.viewer.show_event(ps)

    def do_step(self, ps, dt):
        """

        """
        return self.sia.recurse(ps, dt)


# -- End of File --
