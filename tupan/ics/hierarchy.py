# -*- coding: utf-8 -*-
#

"""
TODO.
"""

from ..particles.system import ParticleSystem


def make_hierarchy(parent, subsys_factory):
    """

    """

    ps = ParticleSystem()
    for p, sf in zip(parent, subsys_factory):
        func, *args = sf
        subsys = func(*args)
        subsys.dynrescale_total_mass(p.total_mass)
        subsys.com_to_origin()
        subsys.com_move_to(p.com_r, p.com_v)
        ps += subsys
    ps.reset_pid()
    ps.com_to_origin()
    ps.scale_to_standard()
    return ps


# -- End of File --
