#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The Simulation class"""

from __future__ import print_function
from pprint import pprint
import random

from .universe import Universe, Body, BlackHole


class IO():
    """  """

    def __init__(self):
        self.myuniverse = Universe()

    def read_data(self, fname):
        print('reading data from \'{0}\''.format(fname))
        for i in xrange(10000): a = i*i-2*i+i/5

        for i in xrange(8):
            self.myuniverse.get_member('body', Body())
        for i in xrange(3):
            self.myuniverse.get_member('bh', BlackHole())
        return self.myuniverse

    def take_a_particle_based_sampling(self):
        pass


class Diagnostic():
    """  """

    def write_diagnostic(self):
        pass


class Simulation(IO):
    """The Simulation class is the top level class for N-body simulations"""

    def __init__(self, args):
        self.time = 0.0           # global simulation time
        self.eta = args.tau
        self.fdia = args.dia      # diagnostic frequency per time unit
        self.fdump = args.out     # indicates how often each particle is dumped
        self.tend = args.end
        self.icfname = args.ic
        self.sim_mode = args.mod


    def prepare_for_run(self):
        pass


    def run(self):
        """Initialize a N-body simulation"""
        print('running...')
        io = IO()
        myuniverse = io.read_data(self.icfname)

        pprint(myuniverse)
        print('running... done')


########## end of file ##########
