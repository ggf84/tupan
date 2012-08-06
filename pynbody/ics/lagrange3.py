#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from ..particles.body import Body


def make_system():
    p = Body(3)

    p.id[:] = [0, 1, 2]
    p.mass[:] = [1.0, 1.0, 1.0]
    p.pos[:] = [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.8660254037844386, 0.0],
               ]
    p.vel[:] = [
                [-0.5, 0.8660254037844386, 0.0],
                [-0.5,-0.8660254037844386, 0.0],
                [ 1.0, 0.0, 0.0],
               ]

    return p


########## end of file ##########
