#!/usr/bin/env python

"""
The Simulation class
"""

from __future__ import print_function
from .universe import Universe


class Simulation():
    """
    The Simulation class is the top level class for N-body simulations
    """

    def __init__(self, args):
        self.args = args
        self.universe = Universe()

    def read_ic(self, filename):
        pass

    def prepare_for_run(self):
        pass

    def write_diagnostic(self):
        pass

    def take_a_particle_based_sampling(self):
        pass

    def run(self):
        """
        Initialize a N-body simulation
        """
        print('running...')
        print(self.args)
        print(self.universe)



########## end of file ##########
