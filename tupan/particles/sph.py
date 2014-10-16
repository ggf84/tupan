# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from .base import AbstractParticle
from ..lib.utils.timing import timings, bind_all


__all__ = ['Sphs']


@bind_all(timings)
class Sphs(AbstractParticle):
    """

    """
    name = None
    default_attr_descr = AbstractParticle.default_attr_descr + [
        ('density', (), 'real', 'density'),
    ]


# -- End of File --
