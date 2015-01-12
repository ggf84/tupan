# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from .base import AbstractParticle, AbstractNbodyMethods
from ..lib.utils.timing import timings, bind_all


__all__ = ['Star']


@bind_all(timings)
class Star(AbstractParticle, AbstractNbodyMethods):
    """

    """
    name = None
    default_attr_descr = AbstractNbodyMethods.default_attr_descr + [
        ('spin', '3, {n}', 'real', 'spin'),
        ('radius', '{n}', 'real', 'radius'),
        ('age', '{n}', 'real', 'age'),
        ('metallicity', '{n}', 'real', 'metallicity'),
    ]


# -- End of File --
