#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Decorators

This module implements useful decorators that can be
used to affect the behaviour of the handlers requested.
"""

from __future__ import print_function
import sys
import time
import types
import atexit
import functools


class EmptyObj(object):
    pass


############
## addmethod
##
def addmethod(instance):
    def wrapper(func):
        func = types.MethodType(func, instance, instance.__class__)
        setattr(instance, func.func_name, func)
        return func
    return wrapper


############
## selftimer
##
VERBOSE_ATCALL = True
VERBOSE_ATEXIT = True
def selftimer(func):
    self = EmptyObj()
    self.ncalls = 0
    self.elapsed = 0
    self.total_elapsed = 0
    self.timer = time.time
    self.func_name = func.__module__ + '.' + func.__name__
    try:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                try:
                    start = self.timer()
                    return func(*args, **kwargs)
                finally:
                    self.ncalls += 1
                    self.elapsed = self.timer() - start
                    self.total_elapsed += self.elapsed
                    if VERBOSE_ATCALL:
                        fmt = '--- {0:s} [call {1}]: {2:g} s'
                        print(fmt.format(self.func_name, self.ncalls,
                                         self.elapsed), file=sys.stderr)
            finally:
                wrapper.selftimer.ncalls = self.ncalls
                wrapper.selftimer.elapsed = self.elapsed
                wrapper.selftimer.total_elapsed = self.total_elapsed
        wrapper.selftimer = EmptyObj()
        wrapper.selftimer.ncalls = self.ncalls
        wrapper.selftimer.elapsed = self.elapsed
        wrapper.selftimer.total_elapsed = self.total_elapsed
        wrapper.selftimer.undecorated = func
        return wrapper
    finally:
        def at_exit():
            if not self.ncalls:
                return
            fmt = ('--- ---\n' +
                   '    {0:s} [after {1} calls]:\n' +
                   '    {2:g} s ({3:g} s per call)')
            print(fmt.format(self.func_name, self.ncalls, self.total_elapsed,
                             self.total_elapsed/self.ncalls), file=sys.stderr)
        if VERBOSE_ATEXIT:
            atexit.register(at_exit)


########## end of file ##########
