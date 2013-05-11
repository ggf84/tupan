# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from ..particles.body import Bodies


def make_system():
    p = Bodies(3)

    p.id = [0, 1, 2]

    p.mass = [1.0, 1.0, 1.0]

    p.rx = [0.0, +1.0, +0.5]
    p.ry = [0.0, 0.0, +0.8660254037844386]
    p.rz = [0.0, 0.0, 0.0]

    p.vx = [-0.5, -0.5, +1.0]
    p.vy = [+0.8660254037844386, -0.8660254037844386, 0.0]
    p.vz = [0.0, 0.0, 0.0]

    return p


########## end of file ##########
