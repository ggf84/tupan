#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The Simulation class"""

from __future__ import print_function
from pprint import pprint

from .models import Plummer
from .particles import (Particles, BlackHoles)


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

        myuniverse = Particles()

        p = Plummer(5471, seed=1)
        p.make_plummer()
        myuniverse.set_member(p.bodies)

        p = Plummer(3, seed=1)
        p.make_plummer()
        bhdata = [tuple(b)+([0.0, 0.0, 0.0],) for b in p.bodies.array.tolist()]
        bh = BlackHoles()
        bh.fromlist(bhdata)
        myuniverse.set_member(bh)


        pprint(myuniverse)
        print('running... done')


########## end of file ##########
