#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from .block import *
from .leapfrog import *

METHS = [LeapFrog, BlockStep]
METH_NAMES = map(lambda m: m.__name__.lower(), METHS)


########## end of file ##########
