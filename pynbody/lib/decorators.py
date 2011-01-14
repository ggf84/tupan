#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import (print_function, with_statement)
import time


class timeit(object):
    def __init__(self, f):
        self.f = f
    def __call__(self, *args, **kwargs):
        tstart = time.time()
        ret = self.f(*args, **kwargs)
        elapsed = time.time() - tstart
        print('time elapsed in <{name}>: {time} s'.format(name=self.f.__name__,
                                                          time=elapsed))
        return ret


########## end of file ##########
