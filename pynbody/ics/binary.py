#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import division
from ..particles.body import Body


def make_binary(m, a, e, m_ratio=1, rcom=[0,0,0], vcom=[0,0,0]):
    """

    """
    if e > 1 or e < 0:
        raise ValueError("eccentricity out of range (0 < e < 1)")

    if not a > 0:
        raise ValueError("semi-major-axis out of range (a > 0)")

    if not m_ratio >= 1:
        raise ValueError("m_ratio out of range (m_ration >= 1)")


    r = a * (1-e)
    v = (m * (2/r - 1/a))**0.5
    m1 = m / (1+m_ratio)
    m2 = m - m1
    r1 =  (m2/m) * r
    r2 = -(m1/m) * r
    v1 =  (m2/m) * v
    v2 = -(m1/m) * v


    b = Body(2)
    b.id[:] = [0, 1]
    b.mass[:] = [m1, m2]
    b.pos[:] = [
                [r1+rcom[0], rcom[1], rcom[2]],
                [r2+rcom[0], rcom[1], rcom[2]],
               ]
    b.vel[:] = [
                [vcom[0], v1+vcom[1], vcom[2]],
                [vcom[0], v2+vcom[1], vcom[2]],
               ]

    return b


########## end of file ##########
