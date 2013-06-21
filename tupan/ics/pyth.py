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

    ps.mass = [3.0, 4.0, 5.0]

    ps.rx = [+1.0, -2.0, +1.0]
    ps.ry = [+3.0, -1.0, -1.0]
    ps.rz = [0.0, 0.0, 0.0]

    ps.vx = [0.0, 0.0, 0.0]
    ps.vy = [0.0, 0.0, 0.0]
    ps.vz = [0.0, 0.0, 0.0]

    ps.id = range(ps.n)
    return ps


########## end of file ##########
