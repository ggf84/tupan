#!/usr/bin/env python

"""
Performs a N-body Simulation
"""

import sys
import argparse
from pynbody import Simulation


def process_cmdline():
    # create the parser
    parser = argparse.ArgumentParser(
        description='Performs a N-body Simulation')

    # add the arguments
    parser.add_argument(
        '--log', type=argparse.FileType('w'), default=sys.stdout,
        help='the file where the log should be written (default: sys.stdout)')

    # parse the command line
    args = parser.parse_args()
    return args


def main(argv=None):
    args = process_cmdline()
    mysim = Simulation(args)
    mysim.run()
    return 0


if __name__ == "__main__":
    status = main()
    sys.exit(status)


########## end of file ##########
