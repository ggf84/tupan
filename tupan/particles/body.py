# -*- coding: utf-8 -*-
#

"""
TODO.
"""

from .base import AbstractParticle, Attributes


class Body(AbstractParticle):
    """

    """
    name = None
    part_type = None
    default_attr_descr = Attributes.default
    extra_attr_descr = Attributes.extra


# -- End of File --
