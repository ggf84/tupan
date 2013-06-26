# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from ..particles.allparticles import ParticleSystem


def make_hierarchy(pps, make_subsys, subsys_size, *args, **kwargs):
    """

    """
    ps = ParticleSystem()
    for p in pps:
        subsys = make_subsys(*args, **kwargs)
        subsys.dynrescale_total_mass(p.mass)
        subsys.dynrescale_size(subsys_size)
        subsys.move_com(p.rcom, p.vcom)
        ps.append(subsys)
    ps.id = range(ps.n)

    return ps


########## end of file ##########
