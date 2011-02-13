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
    """
    Use it to decorate a function you want to measure the execution time.

    It returns the decorated function with an aditional 'selftimer' attribute,
    wich holds the number of calls, the elapsed time in the last call and the
    total elapsed time in all calls of the decorated function. It also adds a
    '__wrapped__' attribute pointing to the original callable function.
    """
    timer = time.time
    func_name = func.__module__ + '.' + func.__name__
    try:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                start = timer()
                return func(*args, **kwargs)
            finally:
                wrapper.selftimer.ncalls += 1
                wrapper.selftimer.elapsed = timer() - start
                wrapper.selftimer.total_elapsed += wrapper.selftimer.elapsed
                if VERBOSE_ATCALL:
                    fmt = '--- {0:s} [call {1}]: {2:g} s'
                    print(fmt.format(func_name, wrapper.selftimer.ncalls,
                                     wrapper.selftimer.elapsed), file=sys.stderr)
        wrapper.selftimer = type('selftimer', (object,),
                                 {'elapsed': 0.0, 'ncalls': 0,
                                  'total_elapsed': 0.0})()
        wrapper.__wrapped__ = func  # adds a __wrapped__ attribute pointing to
                                    # the original callable function. It will
                                    # becomes unnecessary with functools module
                                    # from version 3.2 of python.
        return wrapper
    finally:
        def at_exit():
            if not wrapper.selftimer.ncalls:
                return
            fmt = ('--- ---\n' +
                   '    {0:s} [after {1} calls]:\n' +
                   '    {2:g} s ({3:g} s per call)')
            per_call = wrapper.selftimer.total_elapsed / wrapper.selftimer.ncalls
            print(fmt.format(func_name, wrapper.selftimer.ncalls,
                             wrapper.selftimer.total_elapsed,
                             per_call), file=sys.stderr)
        if VERBOSE_ATEXIT:
            atexit.register(at_exit)


########## end of file ##########
