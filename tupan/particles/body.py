# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from .base import AbstractParticle
from ..lib.utils.timing import timings, bind_all


__all__ = ['Bodies']


@bind_all(timings)
class Bodies(AbstractParticle):
    """

    """
    name = None
    default_attr_descr = AbstractParticle.default_attr_descr + []


# -- End of File --
