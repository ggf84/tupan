# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from ..particles.allparticles import ParticleSystem


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
        subsys.move_com(p.rcom, p.vcom)
        ps.append(subsys)
    ps.id = range(ps.n)

    return ps


########## end of file ##########
