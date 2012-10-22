#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from ..particles.body import Bodies


def make_system():
    p = Bodies(3)

    p.id = [0, 1, 2]

    p.mass = [1.0, 1.0, 1.0]

    p.x = [+0.9700436, -0.9700436,  0.0]
    p.y = [-0.24308753,+0.24308753, 0.0]
    p.z = [ 0.0,        0.0,        0.0]

    p.vx = [+0.466203685,+0.466203685,-0.93240737]
    p.vy = [+0.43236573, +0.43236573, -0.86473146]
    p.vz = [ 0.0,         0.0,         0.0       ]

    return p


########## end of file ##########
