#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The Simulation class"""

from __future__ import print_function
from .io import HDF5IO


class Diagnostic(object):
    """  """

    def write_diagnostic(self):
        pass


class Simulation(object):
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

        io = HDF5IO('input.hdf5')
        myuniverse = io.read_snapshot()

        io = HDF5IO('output.hdf5')
        io.write_snapshot(myuniverse)
        mynewuniverse = io.read_snapshot()

        for i in zip(myuniverse['blackhole'].array, mynewuniverse['blackhole'].array):
            for j in zip(i[0], i[1]):
                print(j[0] == j[1])

        print('running... done')


########## end of file ##########
