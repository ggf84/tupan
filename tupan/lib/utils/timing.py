# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import sys
import timeit
import atexit
import inspect
import functools
from collections import defaultdict


__all__ = ["timings", "bind_all", "Timer"]


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

    def reset_at(self, time):
        self.stopped = False
        self.tic += timeit.default_timer() - time


class MyDefaultdict(defaultdict):
    def __repr__(self):
        import json
        return json.dumps(self, indent=4, sort_keys=True)


def tree():
    return MyDefaultdict(tree)

DTREE = tree()


def timings(meth, cls=None):
    cname = ''
    fname = meth.__name__
    module = meth.__module__
    if cls:
        cname = cls.__name__
        module = cls.__module__

    timer = Timer()
    if cname:
        if '.fget' in fname:
            gname = fname.replace('.fget', '')
            dtree = DTREE[module][cname][gname]['.fget'] = MyDefaultdict(int)
        elif '.fset' in fname:
            sname = fname.replace('.fset', '')
            dtree = DTREE[module][cname][sname]['.fset'] = MyDefaultdict(int)
        elif '.fdel' in fname:
            dname = fname.replace('.fdel', '')
            dtree = DTREE[module][cname][dname]['.fdel'] = MyDefaultdict(int)
        else:
            dtree = DTREE[module][cname][fname] = MyDefaultdict(int)
    else:
        dtree = DTREE[module][fname] = MyDefaultdict(int)

    @functools.wraps(meth)
    def wrapper(*args, **kwargs):
        timer.start()
        ret = meth(*args, **kwargs)
        timer.stop()
        dtree["count"] += 1
        dtree["total"] += timer.elapsed()
        dtree["average"] = dtree["total"] / dtree["count"]
        return ret
    return wrapper


def bind_all(decorator):
    def wrapper(cls):
        for name, meth in inspect.getmembers(cls):
            if inspect.ismethod(meth):
                if inspect.isclass(meth.im_self):
                    # meth is a classmethod
                    meth = meth.im_func
                    setattr(cls, name, classmethod(decorator(meth, cls)))
                else:
                    # meth is a regular method
                    setattr(cls, name, decorator(meth, cls))
            elif inspect.isfunction(meth):
                # meth is a staticmethod
                setattr(cls, name, staticmethod(decorator(meth, cls)))
            elif isinstance(meth, property):
                fget = None
                if meth.fget is not None:
                    meth.fget.__name__ = meth.fget.__name__ + '.fget'
                    fget = decorator(meth.fget, cls)
                fset = None
                if meth.fset is not None:
                    meth.fset.__name__ = meth.fset.__name__ + '.fset'
                    fset = decorator(meth.fset, cls)
                fdel = None
                if meth.fdel is not None:
                    meth.fdel.__name__ = meth.fdel.__name__ + '.fdel'
                    fdel = decorator(meth.fdel, cls)
                doc = meth.__doc__
                setattr(cls, name, property(fget, fset, fdel, doc))
        return cls
    return wrapper


profile = True if "--profile" in sys.argv else False

timings = timings if profile else lambda meth, cls=None: meth
bind_all = bind_all if profile else lambda decor: lambda meth, cls=None: meth

if profile:
    atexit.register(print, DTREE, file=sys.stderr)


# -- End of File --
