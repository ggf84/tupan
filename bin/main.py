#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performs a N-body Simulation.
"""

from __future__ import print_function
import sys
import argparse
from pynbody.simulation import (Simulation, RUN_MODES, METH_NAMES)


def process_cmdline():
    """
    Process command line arguments.
    """
    # create the parser
    parser = argparse.ArgumentParser(description='Performs a N-body Simulation.')

    # add the arguments
    parser.add_argument('-a', '--animate',
                        action='store_true',
                        help='Enable GLviewer animation of the simulation.'
                       )
    parser.add_argument('-d', '--diag_freq',
                        type=int,
                        default=4,
                        help='Diagnostic frequency of the simulation per time '
                             'unit (type: int, default: 4).'
                       )
    parser.add_argument('-e', '--eta',
                        type=float,
                        default=None,
                        required=True,
                        help='Parameter for time step determination (type: flo'
                             'at, default: None).'
                       )
    parser.add_argument('-g', '--gl_freq',
                        type=int,
                        default=128,
                        help='GLviewer event frequency per time unit (type: in'
                             't, default: 128).'
                       )
    parser.add_argument('-i', '--input',
                        type=str,
                        default=None,
                        required=True,
                        help='The file name from which the initial conditions '
                             'must be read (type: str, default: None).'
                       )
    parser.add_argument('-l', '--log_file',
                        type=str,
                        default=sys.stdout,
                        help='The file name where the log should be written (t'
                             'ype: str, default: sys.stdout).'
                       )
    parser.add_argument('-m', '--meth',
                        type=str,
                        default='leapfrog',
                        help='Integration method name {0} (type: str, default:'
                             ' leapfrog).'.format(METH_NAMES)
                       )
    parser.add_argument('-o', '--out',
                        type=int,
                        default=None,
                        help='Number of individual time steps between each par'
                             'ticle output (type: int, default: None).'
                             ' XXX: NOT IMPLEMENTED.'
                       )
    parser.add_argument('-r', '--resdt',
                        type=float,
                        default=1.0,
                        help='Time interval between rewrites of the resumption'
                             ' file. (type: float, default: 1.0).'
                       )
    parser.add_argument('-s', '--smod',
                        type=str,
                        default='newrun',
                        help='Operation mode of the simulation {0} (type: str,'
                             ' default: newrun).'.format(RUN_MODES)
                       )
    parser.add_argument('-t', '--tmax',
                        type=float,
                        default=None,
                        required=True,
                        help='Time to stop the simulation (type: float, defaul'
                             't: None).'
                       )

    # parse the command line
    args = parser.parse_args()

    if args.log_file != sys.stdout:
        # open log-file according to operation mode of the simulation
        if args.smod in RUN_MODES:
            if args.smod == 'restart':
                args.log_file = open(args.log_file, 'a')
            else:
                args.log_file = open(args.log_file, 'w')
        else:
            print('Typo or invalid operating mode for the simulation.')
            print('Available modes:', RUN_MODES)
            print('exiting...')
            sys.exit(1)

    return args


def main():
    """
    The top level main function.
    """
    args = process_cmdline()
    mysim = Simulation(args)
    mysim.evolve()
    return 0


if __name__ == "__main__":
    sys.exit(main())


########## end of file ##########
