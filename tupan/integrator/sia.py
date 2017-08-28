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
def n_drift(ps, dt):
    """Newtonian drift operator.

    """
    for p in ps.members.values():
        if p.n:
            p.time += dt
            p.rdot[0] += p.rdot[1] * dt
    return ps


#
# pn_drift
#
def pn_drift(ps, dt):
    """Post-Newtonian drift operator.

    """
    for p in ps.members.values():
        if p.n:
            p.time += dt
            p.rdot[0] += p.rdot[1] * dt
            p.pn_mr += p.pn_mv * dt
    return ps


#
# n_kick
#
def n_kick(ps, dt):
    """Newtonian kick operator.

    """
    for p in ps.members.values():
        if p.n:
            p.rdot[1] += p.rdot[2] * dt
    return ps


#
# pn_kick
#
def pn_kick(ps, dt, pn=None):
    """Post-Newtonian kick operator.

    """
    #
    ps.set_pnacc(ps, pn=pn)
    for p in ps.members.values():
        if p.n:
            p.pnvel += (p.rdot[2] + p.pnacc) * (dt / 2)

    ps.set_pnacc(ps, pn=pn, use_auxvel=True)
    for p in ps.members.values():
        if p.n:
            pnforce = p.mass * p.pnacc
            p.pn_mv -= pnforce * dt
            p.pn_am -= np.cross(p.rdot[0].T, pnforce.T).T * dt
            p.pn_ke -= (p.pnvel * pnforce).sum(0) * dt
            p.rdot[1] += (p.rdot[2] + p.pnacc) * dt

    ps.set_pnacc(ps, pn=pn)
    for p in ps.members.values():
        if p.n:
            p.pnvel += (p.rdot[2] + p.pnacc) * (dt / 2)
    #
    return ps


#
# sf_n_kick
#
def sf_n_kick(slow, fast, dt):
    """Newtonian slow<->fast kick operator.

    """
    for ps in [slow, fast]:
        for p in ps.members.values():
            if p.n:
                p.rdot[1] += p.rdot[2] * dt
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
        for p in ps.members.values():
            if p.n:
                p.pnvel += (p.rdot[2] + p.pnacc) * (dt / 2)

    slow.set_pnacc(fast, pn=pn, use_auxvel=True)
    for ps in [slow, fast]:
        for p in ps.members.values():
            if p.n:
                pnforce = p.mass * p.pnacc
                p.pn_mv -= pnforce * dt
                p.pn_am -= np.cross(p.rdot[0].T, pnforce.T).T * dt
                p.pn_ke -= (p.pnvel * pnforce).sum(0) * dt
                p.rdot[1] += (p.rdot[2] + p.pnacc) * dt

    slow.set_pnacc(fast, pn=pn)
    for ps in [slow, fast]:
        for p in ps.members.values():
            if p.n:
                p.pnvel += (p.rdot[2] + p.pnacc) * (dt / 2)
    #
    return slow, fast


#
# twobody_solver
#
def twobody_solver(ps, dt, pn=None, kernel=ext.get_kernel('Kepler')):
    """

    """
    if pn:
        raise NotImplementedError('The current version of the '
                                  'Kepler-solver does not include '
                                  'post-Newtonian corrections.')
    else:
        ps0, ps1 = ps[0], ps[1]
        kernel(next(iter(ps0.members.values())),
               next(iter(ps1.members.values())),
               dt=dt)
        ps = ps0 + ps1
        ps.time += dt
    return ps


class SIAXX(object):
    """

    """
    coefs = [(None, None)]

    def __init__(self, manager, meth):
        self.cli = manager.cli
        self.dump = manager.dump
        self.update_tstep = manager.update_tstep
        self.shared_tstep = manager.shared_tstep
        self.evolve = getattr(self, meth)
        self.bridge = getattr(self, 'sf_' + meth)

    def drift(self, ps, dt):
        """Drift operator.

        """
        if self.cli.pn:
            return pn_drift(ps, dt)
        return n_drift(ps, dt)

    def kick(self, ps, dt):
        """Kick operator.

        """
        ps.set_acc(ps)
        if self.cli.pn:
            return pn_kick(ps, dt, pn=self.cli.pn)
        return n_kick(ps, dt)

    def dkd(self, ps, dt):
        """Arbitrary order DKD-type operator.

        """
        kick = self.kick
        drift = self.drift

        if ps.n == 1:
            return drift(ps, dt)

#        if ps.n == 2:
#            return twobody_solver(ps, dt, pn=self.cli.pn)

        coefs = self.coefs
        ck0, cd0 = coefs[-1]

        for ck, cd in coefs[:-1]:
            ps = drift(ps, cd * dt) if cd else ps
            ps = kick(ps, ck * dt) if ck else ps

        ps = drift(ps, cd0 * dt)
        ps = kick(ps, ck0 * dt)
        ps = drift(ps, cd0 * dt)

        for ck, cd in reversed(coefs[:-1]):
            ps = kick(ps, ck * dt) if ck else ps
            ps = drift(ps, cd * dt) if cd else ps

        return ps

    def kdk(self, ps, dt):
        """Arbitrary order KDK-type operator.

        """
        kick = self.kick
        drift = self.drift

        if ps.n == 1:
            return drift(ps, dt)

#        if ps.n == 2:
#            return twobody_solver(ps, dt, pn=self.cli.pn)

        coefs = self.coefs
        cd0, ck0 = coefs[-1]

        for cd, ck in coefs[:-1]:
            ps = kick(ps, ck * dt) if ck else ps
            ps = drift(ps, cd * dt) if cd else ps

        ps = kick(ps, ck0 * dt)
        ps = drift(ps, cd0 * dt)
        ps = kick(ps, ck0 * dt)

        for cd, ck in reversed(coefs[:-1]):
            ps = drift(ps, cd * dt) if cd else ps
            ps = kick(ps, ck * dt) if ck else ps

        return ps

    def recurse(self, ps, dt):
        """

        """
        slow, fast = ps.split_by(lambda obj: obj.tstep > abs(dt))
        ps = self.bridge(slow, fast, dt)
        if self.update_tstep:
            ps.set_tstep(ps, self.cli.eta)
            if self.shared_tstep:
                tstep_min = np.copysign(ps.tstep_min, self.cli.eta)
                for p in ps.members.values():
                    if p.n:
                        p.tstep[...] = tstep_min
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
        if self.cli.pn:
            return sf_pn_kick(slow, fast, dt, pn=self.cli.pn)
        return sf_n_kick(slow, fast, dt)

    def sf_dkd(self, slow, fast, dt):
        """Arbitrary order slow<->fast DKD-type operator.

        """
        if not fast.n:
            slow = self.evolve(slow, dt)
            for s in slow.members.values():
                if s.n:
                    s.nstep += 1
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

        for s in slow.members.values():
            if s.n:
                s.nstep += 1
        self.dump(slow, dt)
        return slow + fast

    def sf_kdk(self, slow, fast, dt):
        """Arbitrary order slow<->fast KDK-type operator.

        """
        if not fast.n:
            slow = self.evolve(slow, dt)
            for s in slow.members.values():
                if s.n:
                    s.nstep += 1
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

        for s in slow.members.values():
            if s.n:
                s.nstep += 1
        self.dump(slow, dt)
        return slow + fast


#
# SIA21
#
class SIA21(SIAXX):
    # REF.: Yoshida, Phys. Lett. A 150 (1990)
    coefs = [(+1.0, +0.5)]


#
# SIA22
#
class SIA22(SIAXX):
    # REF.: Omelyan, Mryglod & Folk, Comput. Phys. Comm. 151 (2003)
    coefs = [(+0.1931833275037836, None),
             (+0.6136333449924328, +0.5)]


#
# SIA43
#
class SIA43(SIAXX):
    # REF.: Yoshida, Phys. Lett. A 150 (1990)
    coefs = [(+1.3512071919596575, +0.67560359597982880),
             (-1.7024143839193150, -0.17560359597982877)]


#
# SIA44
#
class SIA44(SIAXX):
    # REF.: Omelyan, Mryglod & Folk, Comput. Phys. Comm. 151 (2003)
    coefs = [(+0.17861789584480910, None),
             (-0.06626458266981843, +0.71234183106260560),
             (+0.77529337365001860, -0.21234183106260562)]


#
# SIA45
#
class SIA45(SIAXX):
    # REF.: Omelyan, Mryglod & Folk, Comput. Phys. Comm. 151 (2003)
    coefs = [(-0.0844296195070715, +0.27500812123324190),
             (+0.3549000571574260, -0.13479500991067920),
             (+0.4590591246992910, +0.35978688867743724)]


#
# SIA46
#
class SIA46(SIAXX):
    # REF.: Blanes & Moan, J. Comp. Appl. Math. 142 (2002)
    coefs = [(+0.07920369643119570, None),
             (+0.35317290604977400, +0.209515106613362),
             (-0.04206508035771950, -0.143851773179818),
             (+0.21937695575349958, +0.434336666566456)]


#
# SIA67
#
class SIA67(SIAXX):
    # REF.: Yoshida, Phys. Lett. A 150 (1990)
    coefs = [(+0.78451361047755730, +0.39225680523877865),
             (+0.23557321335935813, +0.51004341191845770),
             (-1.17767998417887100, -0.47105338540975644),
             (+1.31518632068391120, +0.06875316825252015)]


#
# SIA69
#
class SIA69(SIAXX):
    # REF.: Kahan & Li, Math. Comput. 66 (1997)
    coefs = [(+0.39103020330868477, +0.19551510165434238),
             (+0.33403728961113600, +0.36253374645991040),
             (-0.70622728118756140, -0.18609499578821270),
             (+0.08187754964805945, -0.31217486576975095),
             (+0.79856447723936240, +0.44022101344371095)]


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

    def __init__(self, ps, cli, *args, **kwargs):
        """

        """
        super(SIA, self).__init__(ps, cli, *args, **kwargs)

        method = cli.method

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

        if self.cli.pn:
            for p in ps.members.values():
                if p.n:
                    p.register_attribute('pn_ke', '{nb}', 'real_t')
                    p.register_attribute('pn_mr', '{nd}, {nb}', 'real_t')
                    p.register_attribute('pn_mv', '{nd}, {nb}', 'real_t')
                    p.register_attribute('pn_am', '{nd}, {nb}', 'real_t')
                    p.register_attribute('pnacc', '{nd}, {nb}', 'real_t')
                    p.register_attribute('pnvel', '{nd}, {nb}', 'real_t')
                    p.pnvel[...] = p.rdot[1]

        if not self.update_tstep:
            for p in ps.members.values():
                if p.n:
                    p.tstep[...] = self.cli.dt_max
        else:
            ps.set_tstep(ps, self.cli.eta)
            if self.shared_tstep:
                tstep_min = np.copysign(ps.tstep_min, self.cli.eta)
                for p in ps.members.values():
                    if p.n:
                        p.tstep[...] = tstep_min

    def dump(self, ps, dt):
        """

        """
        tdiff = abs(self.t_next - ps.global_time)
        ratio = tdiff // abs(dt)
        if ratio > 0:
            if self.viewer:
                if ratio % self.cli.view == 0:
                    self.viewer.show_event(ps)
            if self.dumpper:
                if ratio % self.cli.dump_freq == 0:
                    self.dumpper.append_data(ps)

    def do_step(self, ps, dt):
        """

        """
        return self.sia.recurse(ps, dt)


# -- End of File --
