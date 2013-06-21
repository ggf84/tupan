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

    ps.rx = [+0.9700436, -0.9700436, 0.0]
    ps.ry = [-0.24308753, +0.24308753, 0.0]
    ps.rz = [0.0, 0.0, 0.0]

    ps.vx = [+0.466203685, +0.466203685, -0.93240737]
    ps.vy = [+0.43236573, +0.43236573, -0.86473146]
    ps.vz = [0.0, 0.0, 0.0]

    ps.id = range(ps.n)
    return ps


########## end of file ##########
