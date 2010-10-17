#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The Simulation class
"""

from __future__ import print_function
from .universe import Universe


class IO():
    """
    """
    def read_ic(self, filename):
        pass

    def take_a_particle_based_sampling(self):
        pass


class Diagnostic():
    """
    """
    def write_diagnostic(self):
        pass


class Simulation():
    """
    The Simulation class is the top level class for N-body simulations
    """

    def __init__(self, args):
        self.args = args
        self.myuniv = Universe()

    def prepare_for_run(self):
        pass

    def run(self):
        """
        Initialize a N-body simulation
        """
        print('running...')
        print(self.args)
        print(self.myuniv)


########## end of file ##########
