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
        ('age', '{n}', 'real_t', 'age'),
        ('spin', '3, {n}', 'real_t', 'spin'),
        ('radius', '{n}', 'real_t', 'radius'),
        ('metallicity', '{n}', 'real_t', 'metallicity'),
    ]


# -- End of File --
