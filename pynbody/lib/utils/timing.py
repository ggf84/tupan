#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import sys
import time
import atexit
import functools


__all__ = ['Timer', 'Timing', 'timings']


class Timer(object):
    """

    """
    def __init__(self):
        self.tic = 0.0
        self.toc = 0.0
        self.stopped = False

    def start(self):
        self.stopped = False
        self.tic = time.time()

    def stop(self):
        self.stopped = True
        self.toc = time.time()

    def elapsed(self):
        if not self.stopped:
            self.toc = time.time()
        return self.toc - self.tic


class Timing(object):
    """

    """
    def __init__(self):
        self.timings = {}

    def _collector(self, module, name, elapsed):
        if module in self.timings:
            if name in self.timings[module]:
                self.timings[module][name]["count"] += 1
                self.timings[module][name]["last_call"] = elapsed
                self.timings[module][name]["total"] += elapsed
            else:
                self.timings[module][name] = {}
                self.timings[module][name]["count"] = 1
                self.timings[module][name]["last_call"] = elapsed
                self.timings[module][name]["total"] = elapsed
        else:
            self.timings[module] = {}
            if name in self.timings[module]:
                self.timings[module][name]["count"] += 1
                self.timings[module][name]["last_call"] = elapsed
                self.timings[module][name]["total"] += elapsed
            else:
                self.timings[module][name] = {}
                self.timings[module][name]["count"] = 1
                self.timings[module][name]["last_call"] = elapsed
                self.timings[module][name]["total"] = elapsed

    def __call__(self, func):
        timer = Timer()
        @functools.wraps(func)
        def wrapper(that, *args, **kwargs):
            timer.start()
            ret = func(that, *args, **kwargs)
            timer.stop()
            module = func.__module__
            name = func.__name__
            if hasattr(that.__class__, name):
                name = that.__class__.__name__ + '.' + name
            self._collector(module, name, timer.elapsed())
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


timings = Timing()
atexit.register(print, timings, file=sys.stderr)


########## end of file ##########
