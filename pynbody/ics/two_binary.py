#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import division
from ..particles.body import Body
from .binary import make_binary


def make_binary_from_parent(parent, a, e, m_ratio=1):
    hb = Body()
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
