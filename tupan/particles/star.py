# -*- coding: utf-8 -*-
#

"""
TODO.
"""

from .base import AbstractParticle, AbstractNbodyMethods
from ..lib.utils.timing import timings, bind_all


@bind_all(timings)
class Star(AbstractParticle, AbstractNbodyMethods):
    """

    """
    name = None
    part_type = None
    default_attr_descr = AbstractNbodyMethods.default_attr_descr + [
        ('age', '{nb}', 'real_t', 'age'),
        ('spin', '{nd}, {nb}', 'real_t', 'spin'),
        ('radius', '{nb}', 'real_t', 'radius'),
        ('metallicity', '{nb}', 'real_t', 'metallicity'),
    ]


# -- End of File --
