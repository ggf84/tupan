# -*- coding: utf-8 -*-
#

"""
TODO.
"""

from ..particles.system import ParticleSystem
from ..lib.utils.timing import timings


@timings
def make_hierarchy(parent, make_subsys, *args, **kwargs):
    """

    """
    ps = ParticleSystem()
    for p in parent:
        subsys = make_subsys(*args, **kwargs)
        subsys.dynrescale_total_mass(p.mass)
        subsys.com_to_origin()
        subsys.com_move_to(p.com_r, p.com_v)
        ps.append(subsys)
    ps.reset_pid()
    ps.to_nbody_units()

    return ps


# -- End of File --
