# -*- coding: utf-8 -*-
#

"""
TODO.
"""


import functools
from .timing import timings


def cache(func):
    func = timings(func)
    cache = dict()
    kwmark = object()   # separates positional and keyword args

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = args
        if kwargs:
            key += (kwmark,) + tuple(sorted(kwargs.items()))
        key = hash(key)
        try:
            result = cache[key]
        except:
            if len(cache) > 100:
                cache.clear()
            result = func(*args, **kwargs)
            cache[key] = result
        return result

    return wrapper


def cache_arg(index):
    def cache(func):
        func = timings(func)
        cache = dict()
        kwmark = object()   # separates positional and keyword args

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = args[index]
            if kwargs:
                key += (kwmark,) + tuple(sorted(kwargs.items()))
            key = hash(key)
            try:
                result = cache[key]
            except:
                if len(cache) > 100:
                    cache.clear()
                result = func(*args, **kwargs)
                cache[key] = result
            return result

        return wrapper
    return cache

# cache = cache_arg(slice(None, None, None))


########## end of file ##########
