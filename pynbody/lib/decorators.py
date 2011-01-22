#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import sys
import time


class timeit(object):
    def __init__(self, f):
        self.f = f
    def __call__(self, *args, **kwargs):
        tstart = time.time()
        ret = self.f(*args, **kwargs)
        elapsed = time.time() - tstart
        name = self.f.__module__ + '.' + self.f.__name__
        fmt = 'time elapsed in <{name}>: {time} s'
        print(fmt.format(name=name, time=elapsed), file=sys.stderr)
        return ret


########## end of file ##########
