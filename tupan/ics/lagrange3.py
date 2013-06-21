# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from ..particles.allparticles import ParticleSystem


def make_system():
    """

    """
    ps = ParticleSystem(3)

    ps.mass = [1.0, 1.0, 1.0]

    ps.rx = [0.0, +1.0, +0.5]
    ps.ry = [0.0, 0.0, +0.8660254037844386]
    ps.rz = [0.0, 0.0, 0.0]

    ps.vx = [-0.5, -0.5, +1.0]
    ps.vy = [+0.8660254037844386, -0.8660254037844386, 0.0]
    ps.vz = [0.0, 0.0, 0.0]

    ps.id = range(ps.n)
    return ps


########## end of file ##########
