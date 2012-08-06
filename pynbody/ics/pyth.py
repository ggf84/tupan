#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from ..particles.body import Body


def make_system():
    p = Body(3)

    p.id[:] = [0, 1, 2]
    p.mass[:] = [3.0, 4.0, 5.0]
    p.pos[:] = [
                [ 1.0,  3.0,  0.0],
                [-2.0, -1.0,  0.0],
                [ 1.0, -1.0,  0.0],
               ]
    p.vel[:] = [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
               ]

    return p


########## end of file ##########
