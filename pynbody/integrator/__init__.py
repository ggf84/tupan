#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from . import block
from . import leapfrog


METHS = [leapfrog.LeapFrog, block.BlockStep]
METH_NAMES = map(lambda m: m.__name__.lower(), METHS)


########## end of file ##########
