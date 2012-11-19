#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import division
from ..particles.body import Bodies


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


    b = Bodies(2)

    b.id = [0, 1]

    b.mass = [m1, m2]

    b.x = [r1+rcom[0], r2+rcom[0]]
    b.y = [   rcom[1],    rcom[1]]
    b.z = [   rcom[2],    rcom[2]]

    b.vx = [   vcom[0],    vcom[0]]
    b.vy = [v1+vcom[1], v2+vcom[1]]
    b.vz = [   vcom[2],    vcom[2]]

    return b




def make_binary_from_parent(parent, a, e, m_ratio=1):
    hb = Bodies()
    for child in parent:
        b = make_binary(child.mass, a, e, m_ratio, rcom=child.rcom, vcom=child.vcom)
        hb.append(b)
    hb.id = range(hb.n)
    return hb


def make_hierarchical_binaries(n_levels, a_ratio, m, a, e, m_ratio=1, rcom=[0,0,0], vcom=[0,0,0]):

    parent = make_binary(m, a, e, m_ratio, rcom, vcom)

    level = 0
    while level < n_levels:
        a /= a_ratio
        parent = make_binary_from_parent(parent, a, e, m_ratio)
        level += 1

    return parent


########## end of file ##########
