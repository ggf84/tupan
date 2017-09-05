# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import logging
from .base import Base, power_of_two
from ..lib import extensions as ext


LOGGER = logging.getLogger(__name__)


def sakura_step(ips, jps, dt, flag,
                nforce=2,
                kernel=ext.make_extension('Sakura')):
    """

    """
    consts = kernel.set_consts(dt=dt, flag=flag)

    ibufs = {}
    for i, ip in ips.members.items():
        if ip.n:
            ibufs[i] = kernel.set_bufs(ip, nforce=nforce)
    jbufs = {**ibufs}
    if ips != jps:
        for j, jp in jps.members.items():
            if jp.n:
                jbufs[j] = kernel.set_bufs(jp, nforce=nforce)

    interactions = []
    for i, ip in ips.members.items():
        if ip.n:
            for j, jp in jps.members.items():
                if jp.n:
                    if ip == jp:
                        args = consts + ibufs[i]
                        kernel.triangle(*args)
                    elif (ip, jp) not in interactions:
                        args = consts + ibufs[i] + jbufs[j]
                        kernel.rectangle(*args)
                        interactions.append((jp, ip))

    for i, ip in ips.members.items():
        if ip.n:
            kernel.map_bufs(ibufs[i], ip, nforce=nforce)
            ip.rdot[:2] += ip.drdot
    if ips != jps:
        for j, jp in jps.members.items():
            if jp.n:
                kernel.map_bufs(jbufs[j], jp, nforce=nforce)
                jp.rdot[:2] += jp.drdot


def evolve_sakura(ps, dt):
    """

    """
    sakura_step(ps, ps, dt/2, +1)

    for p in ps.members.values():
        if p.n:
            p.rdot[0] += p.rdot[1] * dt

    sakura_step(ps, ps, dt/2, -1)
    return ps


class Sakura(Base):
    """

    """
    PROVIDED_METHODS = [
        'c.sakura', 'a.sakura',
    ]

    def __init__(self, ps, cli, *args, **kwargs):
        """

        """
        super(Sakura, self).__init__(ps, cli, *args, **kwargs)

        method = cli.method

        if 'c.' in method:
            self.update_tstep = False
            self.shared_tstep = True
        elif 'a.' in method:
            self.update_tstep = True
            self.shared_tstep = True

        for p in ps.members.values():
            if p.n:
                p.register_attribute('drdot', '2, {nd}, {nb}', 'real_t')

        self.e0 = None

    def get_sakura_tstep(self, ps, eta, dt):
        """

        """
        ps.set_tstep(ps, eta)

        for p in ps.members.values():
            if p.n:
                iw_a = 1 / p.tstep_sum
                iw_b = 1 / p.tstep

                diw = (iw_a**0.5 - iw_b**0.5)**2

                p.tstep[...] = 1 / diw

        return power_of_two(ps, dt)

    def do_step(self, ps, dt):
        """

        """
        if self.update_tstep:
            dt = self.get_sakura_tstep(ps, self.cli.eta, dt)
        ps = evolve_sakura(ps, dt)

        for p in ps.members.values():
            if p.n:
                p.time += dt
                p.nstep += 1
        return ps


# -- End of File --
