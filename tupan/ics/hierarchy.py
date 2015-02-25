# -*- coding: utf-8 -*-
#

"""
TODO.
"""

from ..particles import ParticleSystem
from ..lib.utils.timing import timings


@timings
def make_hierarchy(parent_ps, relative_size, make_subsys, *args, **kwargs):
    """

    """
    parent_size = parent_ps.radial_size

    ps = ParticleSystem()
    for p in parent_ps:
        subsys = make_subsys(*args, **kwargs)
        subsys.dynrescale_total_mass(p.mass)
        subsys_size = relative_size * parent_size
        subsys.dynrescale_radial_size(subsys_size)
        subsys.com_to_origin()
        subsys.com_move_to(p.com_r, p.com_v)
        ps.append(subsys)
    ps.pid = range(ps.n)

    return ps


# -- End of File --
