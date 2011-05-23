#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

import numpy as np
from pynbody.lib import _gravity

def get_acc(bi, bj):
    acc = np.empty((len(bi),4), dtype='f8')
    for (i, obj) in enumerate(bi):
        acc[i,:] = _gravity.get_acc(obj['index'].item(),
                                    obj['mass'].item(),
                                    obj['eps2'].item(),
                                    obj['pos'].copy(),
                                    obj['vel'].copy(),
                                    bj.index.copy(),
                                    bj.mass.copy(),
                                    bj.eps2.copy(),
                                    bj.pos.copy(),
                                    bj.vel.copy())
    return acc


########## end of file ##########
