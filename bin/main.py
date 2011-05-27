#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performs a N-body Simulation.
"""

from __future__ import print_function
import sys
import argparse
from pynbody.simulation import (Simulation, METH_NAMES)


def process_cmdline():
    """
    Process command line arguments.
    """
    # create the parser
    parser = argparse.ArgumentParser(description='Performs a N-body Simulation.')

    # add the arguments
    parser.add_argument('--view',
                        action='store_true',
                        help='Enable visualization of the simulation on the fly.'
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
#                        required=True,
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
#                        required=True,
                        help='The file name from which the initial conditions '
                             'must be read (type: str, default: None).'
                       )
    parser.add_argument('--log_file',
                        type=str,
                        default=sys.stdout,
                        help='File name where log messages should be written ('
                             'type: str, default: sys.stdout).'
                       )
    parser.add_argument('--debug_file',
                        type=str,
                        default=sys.stderr,
                        help='File name where error messages should be written'
                             ' (type: str, default: sys.stderr).'
                       )
    parser.add_argument('--debug',
                        action='store_true',
                        help='Enable debug messages.'
                       )
    parser.add_argument('-m', '--meth',
                        type=str,
                        default='leapfrog',
                        choices=METH_NAMES,
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
    parser.add_argument('--restart',
                        action='store_true',
                        help='If enabled, reboots the simulation from the resu'
                             'mption file.'
                       )
    parser.add_argument('-t', '--tmax',
                        type=float,
                        default=None,
#                        required=True,
                        help='Time to stop the simulation (type: float, defaul'
                             't: None).'
                       )

    # parse the command line
    args = parser.parse_args()

    if args.log_file == sys.stdout:
        args.log_file = sys.stdout.name

    if args.debug_file == sys.stderr:
        args.debug_file = sys.stderr.name

    args.fmode = 'w'
    if args.restart:
        args.fmode = 'a'

    return args


def main():
    """
    The top level main function.
    """
    args = process_cmdline()
    if args.restart:
        import pickle
        with open('restart.pickle', 'r') as fobj:
            mysim = pickle.load(fobj)
#        mysim.args.tmax = args.tmax
#        print(mysim.args.tmax)
    else:
        mysim = Simulation(args)
    mysim.evolve()

    return 0


if __name__ == "__main__":
    sys.exit(main())


########## end of file ##########
