# -*- coding: utf-8 -*-
#

"""
TODO.
"""

from .base import AbstractParticle, AbstractNbodyMethods
from ..lib.utils.timing import timings, bind_all


@bind_all(timings)
class Blackhole(AbstractParticle, AbstractNbodyMethods):
    """

    """
    name = None
    part_type = None
    default_attr_descr = AbstractNbodyMethods.default_attr_descr + [
        ('spin', '{nd}, {nb}', 'real_t', 'spin'),
        ('radius', '{nb}', 'real_t', 'radius'),
    ]


# -- End of File --
