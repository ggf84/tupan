# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import timeit


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


# -- End of File --
