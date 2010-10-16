#!/usr/bin/env python

"""
Performs a N-body Simulation
"""

import sys
import argparse
from pynbody import Simulation


def process_cmdline():
    """
    process command line arguments
    """
    # create the parser
    parser = argparse.ArgumentParser(
        description='Performs a N-body Simulation')

    # add the arguments
    parser.add_argument(
        '-d', '--dia', type=int, default=None, help='diagnostic frequency '
        'of the simulation per time unit (default: None)')
    parser.add_argument(
        '-e', '--end', type=float, default=None,
        help='time to stop the simulation (default: None)')
    parser.add_argument(
        '-i', '--ic', default=None, help='the file name from which '
        'the initial conditions must be read (default: None)')
    parser.add_argument(
        '-l', '--log', type=argparse.FileType('w'), default=sys.stdout,
        help='the file where the log should be written (default: sys.stdout)')
    parser.add_argument(
        '-m', '--mod', default='newrun', help='operation mode of '
        'the simulation (newrun/restart) (default: newrun)')
    parser.add_argument(
        '-o', '--out', type=int, default=None, help='number of individual '
        'time steps between each particle output (default: None)')
    parser.add_argument(
        '-t', '--tau', type=float, default=None,
        help='parameter for time step determination (default: None)')

    # parse the command line
    args = parser.parse_args()
    return args


def main():
    """
    The top level main function
    """
    args = process_cmdline()
    mysim = Simulation(args)
    mysim.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())


########## end of file ##########
