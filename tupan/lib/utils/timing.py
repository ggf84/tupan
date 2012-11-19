#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import sys
import timeit
import atexit
import inspect
import functools


__all__ = ["Timer", "decallmethods", "timings"]


class Timer(object):
    """

    """
    def __init__(self):
        self.tic = 0.0
        self.toc = 0.0
        self.stopped = False

    def start(self):
        self.stopped = False
        self.tic = timeit.default_timer()

    def stop(self):
        self.stopped = True
        self.toc = timeit.default_timer()

    def elapsed(self):
        if not self.stopped:
            self.toc = timeit.default_timer()
        return self.toc - self.tic


class Timing(object):
    """

    """
    def __init__(self, is_enabled):
        self.timings = {}
        self.is_enabled = is_enabled


    def collector(self, module, name, elapsed):
        if module in self.timings:
            if name in self.timings[module]:
                self.timings[module][name]["count"] += 1
                self.timings[module][name]["total"] += elapsed
            else:
                self.timings[module][name] = {}
                self.timings[module][name]["count"] = 1
                self.timings[module][name]["total"] = elapsed
        else:
            self.timings[module] = {}
            if name in self.timings[module]:
                self.timings[module][name]["count"] += 1
                self.timings[module][name]["total"] += elapsed
            else:
                self.timings[module][name] = {}
                self.timings[module][name]["count"] = 1
                self.timings[module][name]["total"] = elapsed


    def __call__(self, func):
        if not self.is_enabled:
            return func
        if func.func_code.co_name == "wrapper":
            return func
        timer = Timer()
        @functools.wraps(func)
        def wrapper(that, *args, **kwargs):
            timer.start()
            ret = func(that, *args, **kwargs)
            timer.stop()
            name = func.__name__
            cls = that.__class__
            if hasattr(cls, name):
                name = cls.__name__ + '.' + name
                module = cls.__module__
            else:
                module = func.__module__
            self.collector(module, name, timer.elapsed())
            return ret
        wrapper.__wrapped__ = func  # adds a __wrapped__ attribute pointing to
                                    # the original callable function. It will
                                    # becomes unnecessary with functools module
                                    # from Python v3.2+.
        return wrapper


    def __str__(self):
        mcount = 0
        mark = "+-"
        indent = " "
        mfmt = mark + "{0:s}:"
        nfmt = mark + "{0:s}: [count: {1} | total: {2:.4e} s | average: {3:.4e} s]"
        _str = ""
        for mkey in sorted(self.timings.keys()):
            mcount += 1
            mindent = " " + indent
            _str += "\n" + mindent
            _str += "|\n" + mindent
            _str += mfmt.format(mkey)
            for key in sorted(self.timings[mkey].keys()):
                count = self.timings[mkey][key]["count"]
                total = self.timings[mkey][key]["total"]
                if mcount < len(self.timings):
                    _str += "\n" + mindent + "|" + indent + "|"
                    _str += "\n" + mindent + "|" + indent
                else:
                    _str += "\n" + mindent + " " + indent + "|"
                    _str += "\n" + mindent + " " + indent
                _str += nfmt.format(key, count, total, total / count)
        return mark + "Timings:{0}".format(_str)


def decallmethods(decorator, prefix=''):
    def wrapper(cls):
        for name, meth in inspect.getmembers(cls, inspect.ismethod):
            if name.startswith(prefix):
                setattr(cls, name, decorator(meth))
        return cls
    return wrapper


profile = True if "--profile" in sys.argv else False

timings = Timing(profile)

if profile:
    atexit.register(print, timings, file=sys.stderr)


########## end of file ##########
