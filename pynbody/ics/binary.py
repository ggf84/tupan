#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from ..particles.body import Body


def make_system():
    p = Body(2)

    p.id[:] = [0, 1]
    p.mass[:] = [1.0, 1.0]
    p.pos[:] = [
                [ 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
               ]
    p.vel[:] = [
                [0.0,  0.25, 0.0],
                [0.0, -0.25, 0.0],
               ]

    return p


########## end of file ##########
