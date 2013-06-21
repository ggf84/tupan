# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from ..particles.allparticles import ParticleSystem


def make_system():
    """

    """
    ps = ParticleSystem(4)

    ps.mass = [1.0, 1.0, 1.0, 1.0]

    ps.rx = [+1.382857, 0.0, -1.382857, 0.0]
    ps.ry = [0.0, +0.157030, 0.0, -0.157030]
    ps.rz = [0.0, 0.0, 0.0, 0.0]

    ps.vx = [0.0, +1.871935, 0.0, -1.871935]
    ps.vy = [+0.584873, 0.0, -0.584873, 0.0]
    ps.vz = [0.0, 0.0, 0.0, 0.0]

    ps.id = range(ps.n)
    return ps


########## end of file ##########
