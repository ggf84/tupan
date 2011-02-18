#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import time


indent = ' '*3


class Block(object):
    """

    """

    def __init__(self, eta):
        self.eta = eta
        self.level = [(i, object) for i in range(3)]
        self.depth = len(self.level)

    def distribute_on_block_levels(self, particles):
        pass

    def drift(self, obj):
        """

        """
        print(indent*obj[0], 'D'+str(obj[0])+':', self.eta/2**obj[0])

    def kick(self, obj):
        """

        """
        print(indent*obj[0], 'K'+str(obj[0])+':', self.eta/2**obj[0])

    def force(self, obj):
        """

        """
        print(indent*obj[0], 'F'+str(obj[0]))

    def step(self, idx=0):
        """

        """
        nextidx = idx + 1

        self.drift(self.level[idx])
        self.kick(self.level[idx])

        if (nextidx < self.depth):
            self.step(nextidx)

        self.force(self.level[idx])

        if (nextidx < self.depth):
            self.step(nextidx)

        self.kick(self.level[idx])
        self.drift(self.level[idx])


block = Block(1.0)

t0 = time.time()
block.step()
print('-'*25)
block.step()
elapsed = time.time() - t0
print('step: ', elapsed)



########## end of file ##########
