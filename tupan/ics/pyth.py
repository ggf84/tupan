# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from ..particles.body import Bodies


def make_system():
    p = Bodies(3)

    p.id = [0, 1, 2]

    p.mass = [3.0, 4.0, 5.0]

    p.rx = [+1.0, -2.0, +1.0]
    p.ry = [+3.0, -1.0, -1.0]
    p.rz = [0.0, 0.0, 0.0]

    p.vx = [0.0, 0.0, 0.0]
    p.vy = [0.0, 0.0, 0.0]
    p.vz = [0.0, 0.0, 0.0]

    return p


########## end of file ##########
