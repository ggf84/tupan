# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import division
from ..particles.allparticles import ParticleSystem


def make_binary(m, a, e, m_ratio=1, rcom=[0, 0, 0], vcom=[0, 0, 0]):
    """

    """
    if e > 1 or e < 0:
        raise ValueError("eccentricity out of range (0 < e < 1)")

    if not a > 0:
        raise ValueError("semi-major-axis out of range (a > 0)")

    if not m_ratio >= 1:
        raise ValueError("m_ratio out of range (m_ratio >= 1)")

    r = a * (1 + e)
    v = ((m / a) * (1 - e) / (1 + e))**0.5
    m1 = m / (1 + m_ratio)
    m2 = m - m1
    r1 = (m2 / m) * r
    r2 = -(m1 / m) * r
    v1 = (m2 / m) * v
    v2 = -(m1 / m) * v

    ps = ParticleSystem(2)

    ps.mass = [m1, m2]

    ps.rx = [rcom[0]+r1, rcom[0]+r2]
    ps.ry = [rcom[1], rcom[1]]
    ps.rz = [rcom[2], rcom[2]]

    ps.vx = [vcom[0], vcom[0]]
    ps.vy = [vcom[1]+v1, vcom[1]+v2]
    ps.vz = [vcom[2], vcom[2]]

    ps.id = range(ps.n)
    return ps


def make_binary_from_parent(parent, a, e, m_ratio=1):
    """

    """
    ps = ParticleSystem()
    for child in parent:
        b = make_binary(child.mass, a, e, m_ratio,
                        rcom=child.rcom, vcom=child.vcom)
        ps.append(b)
    ps.id = range(ps.n)
    return ps


def make_hierarchical_binaries(n_levels, a_ratio,
                               m, a, e, m_ratio=1,
                               rcom=[0, 0, 0], vcom=[0, 0, 0]):
    """

    """
    parent = make_binary(m, a, e, m_ratio, rcom, vcom)

    level = 0
    while level < n_levels:
        a /= a_ratio
        parent = make_binary_from_parent(parent, a, e, m_ratio)
        level += 1

    return parent


########## end of file ##########
