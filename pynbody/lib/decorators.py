#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import sys
import time
import types
import atexit


################
## add_method_to
##
def add_method_to(instance):
    def wrapper(func):
        func = types.MethodType(func, instance, instance.__class__)
        setattr(instance, func.func_name, func)
        return func
    return wrapper


############
## selftimer
##
quiet_atcall = False
quiet_atexit = False
class selftimer(object):
    def __init__(self, _quiet_atcall=quiet_atcall, _quiet_atexit=quiet_atexit):
        self.ncalls = 0
        self.elapsed = 0
        self.cumelapsed = 0
        self.quiet_atcall = _quiet_atcall
        if not _quiet_atexit:
            atexit.register(self.atexit)

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            timer = time.time
            self.func = func
            self.ncalls += 1
            try:
                start = timer()
                return func(*args, **kwargs)
            finally:
                self.elapsed = timer() - start
                self.cumelapsed += self.elapsed
                if not self.quiet_atcall:
                    funcname = func.__module__ + '.' + func.__name__
                    fmt = '--- {0:s}: {1:g} s'
                    print(fmt.format(funcname, self.elapsed), file=sys.stderr)
        wrapper.__doc__ = func.__doc__
        wrapper.__name__ = func.__name__
        wrapper.__module__ = func.__module__
        return wrapper

    def atexit(self):
        if not self.ncalls:
            return
        funcname = self.func.__module__ + '.' + self.func.__name__
        fmt = ('--- ---\n' +
               '    {0:s}:\n' +
               '    {1:g} s in {2:d} calls ({3:g} s per call)')
        print(fmt.format(funcname, self.cumelapsed, self.ncalls,
                         self.cumelapsed/self.ncalls), file=sys.stderr)


########## end of file ##########
